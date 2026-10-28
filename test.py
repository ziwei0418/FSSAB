import os
import sys
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from model import FSSAM, FSSAM5s

from util import dataset
from util import transform_new as transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
val_manual_seed = 123
setup_seed(val_manual_seed, True)
seed_array = [321]
val_num = len(seed_array)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/synapse/synapse_split0_1s_batch.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov3_vitb16", choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vit7b16"],
                        help="version of dino")
    parser.add_argument('--episode', help='number of test episodes', type=int, default=1000)
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # ========================================
    # use bfloat16 for the entire program
    # ========================================
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    logger.info(model)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor()
                ])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor()
                ])
        if args.data_set == 'Synapse':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.val_list, transform=val_transform, mode='val',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ========================================
    # Test one batch first to warmup
    # Global autocast needs to cache the conversion of fp32->bfp16
    # ========================================
    validate(val_loader, model, 321, args.episode, warmup=True)

    # ----------------------  VAL  ----------------------
    start_time = time.time()
    mDice_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    class_iou_class_all = []  # 存储所有seed的每个类别IOU
    class_dice_class_all = []

    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
        class_miou, class_mdice, class_iou_class, class_dice_class = validate(val_loader, model, val_seed, args.episode)
        mDice_array[val_id], mIoU_array[val_id] = class_miou, class_mdice
        class_iou_class_all.append(class_iou_class)
        class_dice_class_all.append(class_dice_class)

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('mDice: {}'.format(np.round(mDice_array, 4)))

    for i in range(len(class_iou_class_all)):
        class_iou_list = class_iou_class_all[i]
        class_dice_list = class_dice_class_all[i]
        for j in range(len(class_iou_list)):      
            label_name = args.LABEL_LIST[args.split][j]
            logger.info('Val_{} Result: iou {:.4f}.'.format(label_name, class_iou_list[j]))
            logger.info('Val_{} Result: dice {:.4f}.'.format(label_name, class_dice_list[j]))   

    print('-' * 43)
    print('Best_Seed_mIou: {} \t Best_Seed_mDice: {}'.format(seed_array[mIoU_array.argmax()],
                                                                         seed_array[mDice_array.argmax()]))
    print('Best_mIoU: {:.4f} \t Best_mDice: {:.4f}'.format(
        mIoU_array.max(), mDice_array.max()))
    print('Mean_mIoU: {:.4f} \t  Mean_mDice: {:.4f}'.format(
        mIoU_array.mean(), mDice_array.mean()))


def validate(val_loader, model, val_seed, episode, warmup=False):
    if not warmup:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    
    intersection_meter = AverageMeter()  # 用于计算，更新交集
    union_meter = AverageMeter()  # 用于计算，更新并集
    target_meter = AverageMeter()  # 用于计算，更新真实目标像素
    output_meter = AverageMeter()  # 用于计算，更新预测目标像素 

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'Synapse':
        test_num = 1000
        split_gap = 2

    class_intersection_meter = [0] * split_gap  # 用于后续存储 每个类别的交集
    class_union_meter = [0] * split_gap  # 用于后续存储 每个类别的并集
    class_target_meter = [0] * split_gap  # 用于后续存储 每个类别的真实目标像素
    class_output_meter = [0] * split_gap  # 用于后续存储 每个类别的预测目标像素

    setup_seed(val_seed, args.seed_deterministic)

    # 定义损失函数为交叉熵损失
    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if iter_num == 1 and warmup: break

            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            # move to GPU
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)

            priors = None

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    start_time = time.time()
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize: # 使用原始label来评估
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

                     # 计算损失loss
                    output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
                    output = output.squeeze(0)
                    label = target.clone()
                    label[label == 255] = 0
                    loss = criterion(output, label.float())
            
            # 二值化output
            output = torch.sigmoid(output)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            subcls = subcls[0].cpu().numpy()[0]

            # 计算 交集，并集，真实目标像素
            intersection, union, area_output, area_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, area_output, area_target = intersection.cpu().numpy(), union.cpu().numpy(), area_output.cpu().numpy(), area_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), output_meter.update(area_output), target_meter.update(area_target)
             # 每个类别的 交集，并集, 预测掩码数，真实掩码数
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]
            class_output_meter[subcls] += area_output[1]
            class_target_meter[subcls] += area_target[1]

            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            dice = (2 * intersection_meter.val[1]) / (output_meter.val[1] + target_meter.val[1] + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            remain_iter = test_num / args.batch_size_val - iter_num
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            # 周期性的报告机制， 打印当前的评估状态（对应 1%、2%、3%... 的进度）
            if ((i + 1) % round((test_num / 100)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Dice {dice:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              remain_time=remain_time,
                                                              loss_meter=loss_meter,
                                                              dice=dice))
    if not warmup:
        val_time = time.time() - val_start


        class_iou_class = []  # 存储每个类别的IOU
        class_miou = 0
        class_dice_class = [] # 存储每个类别的DICE
        class_mdice = 0
        for i in range(len(class_intersection_meter)):
            # 依次遍历验证到的所有类
            class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou

            # 计算每个类别的DICE
            class_dice = (2 * class_intersection_meter[i]) / (class_output_meter[i] + class_target_meter[i] + 1e-10)
            class_dice_class.append(class_dice)
            class_mdice += class_dice

        # 按所有类别平均的IOU 和 dice
        class_miou = class_miou * 1.0 / len(class_intersection_meter) 
        class_mdice = class_mdice * 1.0 / len(class_intersection_meter)
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))  # final
        logger.info('meanDICE---Val result: mDICE {:.4f}.'.format(class_mdice))  # final

        logger.info('<<<<<<< Novel Results <<<<<<<')
        # 按类别打印IOU  DICE
        for i in range(split_gap):
            label_name = args.LABEL_LIST[args.split][i]
            logger.info('Val_{} Result: iou {:.4f}.'.format(label_name, class_iou_class[i]))
            logger.info('Val_{} Result: dice {:.4f}.'.format(label_name, class_dice_class[i]))
            
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

        return class_miou, class_mdice, class_iou_class, class_dice_class


if __name__ == '__main__':
    main()
