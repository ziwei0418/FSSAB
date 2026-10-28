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
from pathlib import Path

from util import dataset
from util import transform_new as transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')  #
    parser.add_argument('--viz', action='store_true', default=True)
    parser.add_argument('--config', type=str, default='config/synapse/synapse_split0_1s_batch.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov3_vitb16", choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vit7b16"],
                        help="version of dino")
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
    # Create model and optimizer
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    # Freeze backbone
    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model)

    # Initialize process for distributed training
    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # ========================================
    # use bfloat16 for the entire program
    # ========================================
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))


def main():
    # 初始化 args, logger, writer
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    # 创建模型和优化器，得到 model, optimizer
    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    if main_process():
        logger.info(model)
    if main_process() and args.viz:  # 是否用tensorborad
        writer = SummaryWriter(args.result_path)  # 指定tensorboard保存路径
    # 将模型写入tensorboard
    # writer.add_graph(model, 
    #                  (torch.randn(args.batch_size, 3, args.train_h, args.train_w).cuda(),
    #                     torch.randn(args.batch_size*args.shot, 1, args.train_h, args.train_w).cuda(),
    #                     torch.randn(args.batch_size, args.shot, args.train_h, args.train_w).cuda(),
    #                     torch.randn(args.batch_size, args.train_h, args.train_w).cuda()
    #                  ))

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Train
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor()
        ])
    if args.data_set == 'Synapse':
       train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.train_list, transform=train_transform, mode='train',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                     filter_intersection=args.filter_intersection)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True,
                                               shuffle=False if args.distributed else True)
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
                                       ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                       filter_intersection=args.filter_intersection)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ----------------------  TRAINVAL  ----------------------
    global best_miou ,best_mdice, best_iou_fg, best_dice_fg, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    best_miou = 0.
    best_mdice = 0.
    best_iou_fg = 0.
    best_dice_fg = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0
    best_miou_m = 0.
    best_miou_b = 0.
    best_FBiou_m = 0.

    start_time = time.time()
    
    # ========================================
    # Test one batch first to warmup
    # Global autocast needs to cache the conversion of fp32->bfp16
    # ========================================
    validate(val_loader, model, warmup=True)
    print('==== Warmup Done! ====')

    # amp
    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        # early stop - 75 epochs
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ----------------------  TRAIN  ----------------------
        loss_train, mIoU_train, mDice_train = train(train_loader, val_loader, model, optimizer, epoch, scaler)

        if main_process() and args.viz:
            writer.add_scalar('mDice_train', mDice_train, epoch_log)
            writer.add_scalar('loss_train', loss_train, epoch_log)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch % 1 == 0:
            # loss_val, FBIoU, mIoU, pIoU = validate(val_loader, model)
            loss_val, class_miou, class_mdice = validate(val_loader, model)
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('class_mdice_val', class_mdice, epoch_log)

            # save model for <testing>
            if class_mdice > best_mdice:
                best_miou, best_mdice, best_epoch = class_miou, class_mdice, epoch
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_mdice) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch) + '_{:.4f}'.format(
                        best_mdice) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               filename)
            
        if args.debug:
            break

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.arch + '\t Group:{} \t Best_step:{}'.format(args.split, best_epoch))
        print('mIoU:{:.4f}'.format(best_miou))
        print('mDice:{:.4f}'.format(best_mdice))
        print('>' * 80)
        print('%s' % datetime.datetime.now())

# 可视化 image，label，output
def visualize(input, target, output, epoch, i):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # 自定义颜色映射：黑（背景）、红（前景）、灰（忽略）
    cmap = mcolors.ListedColormap(['black', 'yellow', 'gray'])
    plt.subplot(1,3,1)
    plt.title("Image")
    tmp_input = input[0]
    tmp_input[tmp_input == 255] = 2
    tmp_input = (tmp_input - tmp_input.min()) / (tmp_input.max() - tmp_input.min() + 1e-10)
    plt.imshow(tmp_input.permute(1, 2, 0).cpu().numpy(), cmap=cmap, vmin=0, vmax=2)

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    tmp_target = target[0]
    tmp_target[tmp_target == 255] = 2
    plt.imshow(tmp_target.cpu().numpy(), cmap=cmap, vmin=0, vmax=2)

    plt.subplot(1,3,3)
    plt.title("Prediction")
    tmp_output = output[0]
    tmp_output[tmp_output == 255] = 2
    plt.imshow(tmp_output.cpu().detach().numpy(), cmap=cmap, vmin=0, vmax=2)
    plt.show()
    filename = f"epoch{epoch+1}_{i+1}.png"
    plt.savefig("/autodl-fs/data/train_res/" + filename)


def train(train_loader, val_loader, model, optimizer, epoch, scaler):
    global best_mdice, best_miou, best_piou, best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter1 = AverageMeter()
    aux_loss_meter2 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    output_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if args.fix_bn:
        model.apply(fix_bn)  # fix batchnorm

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    if main_process():
        print('Warmup: {}'.format(args.warmup))
    # subcls中是所选训练类别在self.sub_list中的索引
    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast():
            output, main_loss, aux_loss1, aux_loss2 = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls)
            loss = main_loss # 加权Dice损失

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        n = input.size(0)  # batch_size

        output = torch.sigmoid(output)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        # visualize(input, target, output, epoch, i)

        # 计算 交集，并集，预测目标像素 ，真实目标像素
        intersection, union, area_output, area_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, area_output, area_target = intersection.cpu().numpy(), union.cpu().numpy(), area_output.cpu().numpy(), area_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), output_meter.update(area_output), target_meter.update(area_target)

        # accuracy 所有预测正确的像素数 / 所有真实像素数
        # K=2, 0:背景, 1:前景   若要算总的，就是sum(intersection_meter.val)
        # 916/916    output_meter.val[1] 78169  accuracy包含前景背景，医学图像分割前景太小，没有什么意义
        accuracy = (sum(intersection_meter.val)) / (sum(target_meter.val) + 1e-10)  # allAcc
        dice_fg = (2 * intersection_meter.val[1]) / (output_meter.val[1] + target_meter.val[1]+ 1e-10)
        iou_fg = intersection_meter.val[1] / (union_meter.val[1] + 1e-10)
        print(f"epoch:{epoch}-{i},前景Dice: {dice_fg:.4f}, 前景IoU: {iou_fg:.4f}, accuracy: {accuracy:.4f}")


        main_loss_meter.update(main_loss.item(), n)  # 主要损失
        # aux_loss_meter1.update(aux_loss1.item(), n)
        # aux_loss_meter2.update(aux_loss2.item(), n)
        loss_meter.update(loss.item(), n)  # 总损失

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss1 {aux_loss_meter1.val:.4f} '
                        'AuxLoss2 {aux_loss_meter2.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'
                        'Dice_fg {dice_fg:.4f}. '
                        'IOU_fg {iou_fg:.4f}'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter1=aux_loss_meter1,
                                                          aux_loss_meter2=aux_loss_meter2,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy,
                                                          dice_fg=dice_fg,
                                                          iou_fg=iou_fg))
            if args.viz:
                writer.add_scalar('loss_train', loss_meter.val, current_iter)
                writer.add_scalar('loss_train_main', main_loss_meter.val, current_iter)
                
        # ---------- SubEpoch VAL 在训练中途（半个 epoch）执行验证 -----------
        if args.evaluate and args.SubEpoch_val and ((args.epochs <= 100 and epoch % 1 == 0) or (epoch > 100)) and (i == round(len(train_loader) / 2)):  # <if> max_epoch<=100 <do> half_epoch Val
            # loss_meter.avg, mIoU, class_miou, iou_class[1]
            # loss_val, FBIoU, mIoU, pIoU = validate(val_loader, model)
            loss, class_miou, class_mdice = validate(val_loader, model)
            val_num += 1
            # 根据 Dice 选出最好模型并保存 checkpoint
            if class_mdice > best_mdice:
                best_miou, best_mdice, best_epoch = class_miou, class_mdice, (epoch - 0.5)
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch - 0.5) + '_{:.4f}'.format(
                        best_mdice) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(
                        epoch - 0.5) + '_{:.4f}'.format(best_mdice) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch - 0.5, 
                         'state_dict': model.state_dict(), 
                         'optimizer': optimizer.state_dict()
                         },filename)

            model.train()
            if args.fix_bn:
                model.apply(fix_bn)  # fix batchnorm

        if args.debug and i==10: # for quick test
            break
    
    # 前景的iou,dice
    iou_fg = intersection_meter.sum[1] / (union_meter.sum[1] + 1e-10)
    dice_fg = (2 * intersection_meter.sum[1]) / (output_meter.sum[1] + target_meter.sum[1] + 1e-10)
    accuracy_class = intersection_meter.sum[1] / (target_meter.sum[1] + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: iou_fg/dice_fg/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, iou_fg,
                                                                                           dice_fg, mAcc, allAcc))

    return main_loss_meter.avg, iou_fg, dice_fg


def validate(val_loader, model, warmup=False):
    if main_process() and not warmup:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()  # 当前batch的时间
    model_time = AverageMeter()  # 模型计算的时间
    data_time = AverageMeter()
    loss_meter = AverageMeter()  # 用于计算，更新损失

    intersection_meter = AverageMeter()  # 用于计算，更新交集
    union_meter = AverageMeter()  # 用于计算，更新并集
    target_meter = AverageMeter()  # 用于计算，更新真实目标像素
    output_meter = AverageMeter()  # 用于计算，更新预测目标像素

    if args.data_set == 'pascal':
        test_num = 1000  # 测试次数
        split_gap = 5  # 是每个fold的类别数
    elif args.data_set == 'Synapse':
        test_num = 200
        split_gap = 2

    class_intersection_meter = [0] * split_gap  # 用于后续存储 每个类别的交集
    class_union_meter = [0] * split_gap  # 用于后续存储 每个类别的并集
    class_target_meter = [0] * split_gap  # 用于后续存储 每个类别的真实目标像素
    class_output_meter = [0] * split_gap  # 用于后续存储 每个类别的预测目标像素

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    # 定义损失函数为交叉熵损失
    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    # 得到训练轮次
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    # ========================================
    # Saving priors
    # ========================================
    prior_dir = "priors/"
    prior_dir = "{}{}/".format(prior_dir, args.data_set)  # priors/pascal/
    prior_dir = "{}{}/".format(prior_dir, args.sam2_type)  # priors/pascal/small/
    prior_dir = "{}{}/".format(prior_dir, args.ver_dino)  # priors/pascal/small/dinov2_vitb14/
    prior_dir = "{}split{}/".format(prior_dir, args.split)  # priors/pascal/small/dinov2_vitb14/split0/
    prior_dir = "{}{}shot/".format(prior_dir, args.shot)  # priors/pascal/small/dinov2_vitb14/split0/1shot

    # for e in range(db_epoch):  # 验证多轮
    for e in range(db_epoch):  # for debug
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            # input, target -->  query,   ori_label query的原始label，用于评估
            # s_input, s_mask --> support
            # subcls --> 当前类别
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

            # 加载 prior masks
            prior_path = "{}{}.npy".format(prior_dir, iter_num)
            if os.path.exists(prior_path) and not warmup:
                priors = np.load(prior_path)
                priors = torch.from_numpy(priors)
                priors = priors.cuda(non_blocking=True)
            else:
                priors = None

            start_time = time.time()
            with torch.no_grad():
                with autocast():
                    # 得到结果
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize:  # 使用原始label来评估
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

                    # 计算损失loss
                    output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
                    output = output.squeeze(0)
                    label = target.clone()
                    label[label == 255] = 0
                    loss = criterion(output, label.float())

            # 保存 prior masks
            if main_process():
                if not os.path.exists(prior_path):
                    # 如果上级目录不存在就创建
                    Path(prior_path).parent.mkdir(parents=True, exist_ok=True)
                    priors = priors.detach().cpu().numpy()
                    np.save(prior_path, priors)

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

            # 周期性的报告机制， 打印当前的评估状态（对应 1%、2%、3%... 的进度）
            if ((i + 1) % round((test_num / 100)) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Dice {dice:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              dice=dice))
            if args.debug and i==10: # for quick test
                break

    if not warmup:
        val_time = time.time() - val_start

        # 整个验证集的总体 前景 Dice， iou    Global Dice（全局前景 Dice）
        # iou_fg = intersection_meter.sum[1] / (union_meter.sum[1] + 1e-10)
        # dice_fg = (2 * intersection_meter.sum[1]) / (output_meter.sum[1] + target_meter.sum[1] + 1e-10)

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

        if main_process():
            logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))  # final
            logger.info('meanDICE---Val result: mDICE {:.4f}.'.format(class_mdice))  # final

            logger.info('<<<<<<< Novel Results <<<<<<<')
            # 按类别打印IOU  DICE
            for i in range(split_gap): # split_gap是每个fold的类别数
                label_name = args.LABEL_LIST[args.split][i]
                logger.info('Val_{} Result: iou {:.4f}.'.format(label_name, class_iou_class[i]))
                logger.info('Val_{} Result: dice {:.4f}.'.format(label_name, class_dice_class[i]))
            
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

            print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

        # 返回 平均损失， mIOU：FBIoU， 
        # class_miou, class_mdice: 按所有类别平均的iou和dice
        # iou_class[1]: 是前景IOU
        return loss_meter.avg, class_miou, class_mdice


if __name__ == '__main__':
    main()
