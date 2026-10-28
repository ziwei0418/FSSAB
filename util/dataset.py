import sys
import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

"""读取全部的train.txt文件，根据sub_list，筛选出符合条件的训练集图像路径"""
def make_dataset_synapse(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    
    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []
    list_read = open(data_list).readlines()  # 读取所有的训练集路径train.txt
    print("Processing data...{}".format(sub_list))  # sub_list是当前fold的可训练类别
    # 初始化字典，存储每个可训练的子类对应的文件
    sub_class_file_list = {} 
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []
    
    for l_idx in tqdm(range(len(list_read))):  # 遍历train_list中的所有训练集样本
        line = list_read[l_idx]
        line = line.strip() # 去除每行首尾的空白字符  case0009_slice000
        line_path = os.path.join(data_root, line + ".npz")  # 拼接出 .npz 文件的完整路径
        line_data = np.load(line_path)
        label = line_data['label']   # 标签矩阵
        label_class = np.unique(label).tolist() # 找出标签中的所有类别
        print(line + ":" + str(label_class))

        # 去除背景类或无效标记：
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        if len(label_class) == 0:
            continue

        new_label_class = []

        # 筛选符合条件的标签类别，存入new_label_class
        # 根据filter_intersection，设置训练集中是否包含测试集样本 setting2
        if filter_intersection:  # 去除 filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)): # 判断 label_class 中的所有元素是否都包含在 sub_list
                # 如果上述条件不满足，则说明当前图片中包含测试类别，后面label_class=[]，这张图片就不要了
                for c in label_class:
                    if c in sub_list:
                        new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    new_label_class.append(c)

        label_class = new_label_class  # 符合条件的类别

        if len(label_class) > 0: # 如果=0说明 背景包含了测试类别，这张图片就不要了
            image_label_list.append(line_path)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(line_path)

    print("Checking image&label pair {} list done! ".format(split))
    # 返回的image_label_list，sub_class_file_list 内部是 npz 文件路径
    return image_label_list, sub_class_file_list

"""
data_root： 数据集根目录
data_list：是train_list，包含所有训练集样本路径的txt文件
sub_list： 当前fold的训练类别 列表
filter_intersection: setting设置，用来控制训练集是否包含测试集类别的图像
返回：
     image_label_list 训练集图像集合, sub_class_file_list 按类别划分的训练集图像集合（一张图片可以出现在多个类别中）
"""
def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    data_set_name = data_root.split('/')[-2]
    if data_set_name == 'Synapse':
        image_label_list, sub_class_file_list = make_dataset_synapse(split, data_root, data_list, sub_list, filter_intersection) 
        return image_label_list, sub_class_file_list   

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []
    list_read = open(data_list).readlines()  # 读取所有的训练集路径
    print("Processing data...".format(sub_list))  # sub_list是当前fold的可训练类别
    # 初始化字典，存储每个可训练的子类对应的文件
    sub_class_file_list = {} 
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []
    
    for l_idx in tqdm(range(len(list_read))):  # 遍历train_list中的所有训练集样本
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])  # 图片名称
        label_name = os.path.join(data_root, line_split[1])  # 标签名称
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()  # 获取标签中包含的所有类别

        # 移除0和255
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []

        # 筛选符合条件的标签类别，存入new_label_class
        # 根据filter_intersection，设置训练集中是否包含测试集样本
        if filter_intersection:  # 去除 filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)): # 判断 label_class 中的所有元素是否都包含在 sub_list
                # 如果上述条件不满足，则说明当前图片中包含测试类别，后面label_class=[]，这张图片就不要了
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0], target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:
                            new_label_class.append(c)
        else:  # false  # 不去除，直接筛选出符合条件的类别
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class  # 符合条件的类别

        if len(label_class) > 0: # 如果=0说明 背景包含了测试类别，这张图片就不要了
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    """
    data_list: 所有训练集样本的路径文件
    data_set: 数据集名称
    """
    def __init__(self, split=3, shot=1, data_root=None, base_data_root=None, data_list=None, data_set=None,
                 use_split_coco=False,
                 transform=None, transform_tri=None, mode='train', ann_type='mask',
                 ft_transform=None, ft_aug_size=None, filter_intersection=False):

        assert mode in ['train', 'val', 'demo', 'finetune']
        assert data_set in ['Synapse', 'coco']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        # 1. 根据数据集设置类别数
        if data_set == 'Synapse':
            self.num_classes = 8   # 共13个类，选择测试4个类 left kidney, right kidney, liver, and spleen 
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split  # fold 数
        self.shot = shot    # 支持集数量
        self.data_root = data_root
        self.base_data_root = base_data_root
        self.ann_type = ann_type  # mask，提示类型
        self.data_set = data_set

        # 按类别分割 交叉验证子集 2折，上腹部一折，下腹部一折
        if data_set == 'Synapse':
            # 1 aorta, 2 gallbladder, 3 left kidneyspleen, 4 right kidney, 
            # 5 liver, 6 pancreas, 7 Slpeen, 8 stomach
            self.class_list = [1,2,3,4,5,6,7,8]  
            # class mapping:
            # 1 aorta, 2 gallbladder, 3 left kidney, 4 right kidney,
            # 5 liver, 6 pancreas, 7 spleen, 8 stomach
            if self.split == 1:  # 上腹部训练 liver 5 and spleen 7
                self.sub_list = [5, 7, 1, 2, 6, 8]  # 训练类别
                self.sub_val_list = [3, 4]    # 验证类别
            elif self.split == 0: # 下腹部训练 left kidney 3 and right kidney 4
                self.sub_list = [3, 4, 1, 2, 6, 8]
                self.sub_val_list = [5, 7]
            elif self.split == 2:  # 前胸/上中部训练 aorta 1 and gallbladder 2
                self.sub_list = [1, 2, 3, 4, 5, 7]
                self.sub_val_list = [6, 8]  # 验证类别 pancreas 6 and stomach 8
            elif self.split == 3:  # 对应的互补折
                self.sub_list = [6, 8, 3, 4, 5, 7]
                self.sub_val_list = [1, 2]

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        # # @@@ For convenience, we skip the step of building datasets and instead use the pre-generated lists @@@
        # if self.mode == 'train':
        #     # 按照sub_list，筛选出对应测试类别的训练集图像路径
        #     train_data_root = data_root + "/train_npz"
        #     self.data_list, self.sub_class_file_list = make_dataset(split, train_data_root, data_list, self.sub_list, filter_intersection)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        # elif self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
        #     # 按照sub_val_list，筛选出对应验证类别的图像路径
        #     val_data_root = data_root + "/train_npz"
        #     self.data_list, self.sub_class_file_list = make_dataset(split, val_data_root, data_list, self.sub_val_list, False)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 

        mode = 'train' if self.mode == 'train' else 'val'

        # fss_list_root 根据mode，取值

        if filter_intersection:
            fss_list_root = './lists/{}/all_class/setting2/{}/'.format(data_set, mode)
        else:
            fss_list_root = './lists/{}/all_class/setting1/{}/'.format(data_set, mode)
        # 根据split取第几折的文件，对应要处理的图片路径
        # fss_data_list_path 是所有的训练集，在一起
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
        # fss_sub_class_file_list_path 是一个字典，按类别分类的所有训练集
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        # # Write FSS Data，写入文件中
        # if not os.path.exists(fss_data_list_path):
        #     os.makedirs(os.path.dirname(fss_data_list_path), exist_ok=True)
        # if not os.path.exists(fss_sub_class_file_list_path):
        #     os.makedirs(os.path.dirname(fss_sub_class_file_list_path), exist_ok=True)
        # with open(fss_data_list_path, 'w') as f:
        #     for item in self.data_list:
        #         f.write(item + '\n')
        # with open(fss_sub_class_file_list_path, 'w') as f:
        #     f.write(str(self.sub_class_file_list))

        # fss_data_list_path读取train集的img和mask，存储到self.data_list
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = [] # 存储所有符合条件的训练集 npz文件
        for line in f_str:
            self.data_list.append(line.strip())  # 存储了训练集的img和label路径
        
        # fss_sub_class_file_list_path读取按类别分类的img和mask，存储到self.sub_class_file_list
        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        self.sub_class_file_list = eval(f_str)  # 字典，按类别分类的训练集

        self.transform = transform  # 数据增强
        # self.transform_tri = transform_tri  # 无
        # self.ft_transform = ft_transform  # 无
        # self.ft_aug_size = ft_aug_size  # 无
 
    def __len__(self):
        return len(self.data_list)
    
    def getitem_synapse(self, index):
        # ====
        # 获取训练image和label
        npz_path = self.data_list[index] # self.data_list存储了训练集的img和label路径
        with np.load(npz_path) as data:
            image = data['image']  # [512, 512]
            label = data['label']  # [512, 512]

        image = np.stack([image]*3, axis=-1)  # 转为3通道 [512,512,3]
        image = image.astype(np.float32)

        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + npz_path + "\n"))
        # 获取image包含的所有类别
        label_class = np.unique(label).tolist()
        if 0 in label_class: # 去除背景类别
            label_class.remove(0)
        if 255 in label_class:  # 去除忽略类别
            label_class.remove(255)
        
        # 筛选有效类别
        # 根据运行模式，筛选出当前图像中属于有效类别
        new_label_class = []
        for c in label_class: # 遍历image包含的所有类别
            # 如果在验证集类别中
            if c in self.sub_val_list:  # sub_val_list是验证集类别
                if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
                    new_label_class.append(c)
            # 如果在训练集类别中
            if c in self.sub_list:  # sub_list是训练集类别
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class  # [5,7] / [3,4]
        assert len(label_class) > 0

        # 随机选择一个类别，作为当前任务的目标类别
        class_chosen = label_class[random.randint(1, len(label_class)) - 1]
        # 把原始多类标签图，转换为“单类 + 忽略区域”的二值掩码（binary mask）
        target_pix = np.where(label == class_chosen)  # 找出 label 中 值等于目标类别 (class_chosen) 的所有像素位置
        ignore_pix = np.where(label == 255) # # 找出 需要忽略的像素（label 值是 255）的位置
        label[:, :] = 0  # 将整张标签图重置为 0。
        if target_pix[0].shape[0] > 0:  # 就把目标类别对应的像素标为 1
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255  # 其他都为0

        # 在sub_class_file_list中，取出对应class_chosen类别的所有图像
        # file_class_chosen:包含所选类别对应的所有图像
        file_class_chosen = self.sub_class_file_list[class_chosen]  # sub_class_file_list是验证集的文件
        num_file = len(file_class_chosen)  # 该类别对应的图像数量

        # 获得support的 image和label的路径
        support_npz_path_list = []  # 存放选取的support图像路径 
        support_idx_list = []  # 存放选取的support图像索引
        # 随机选择shot个支持集图像
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_npz_path = npz_path
            # 避免support图像重复，避免support和query是一张图像
            while ((support_npz_path == npz_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_npz_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_npz_path_list.append(support_npz_path)

        subcls_list = [] # 存储当前选中类别在self.sub_list中的索引
        if self.mode == 'train':
            # 如果 self.sub_list 是 [1, 5, 10, 15] 且 class_chosen 是 10，那么 index() 返回 2。
            subcls_list.append(self.sub_list.index(class_chosen))  # self.sub_list是当前fold的训练类别
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))

        # 加载support图像，并进行预处理
        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        for k in range(self.shot):
            # 获取image和label的路径
            support_npz_path = support_npz_path_list[k]
            # 读取image和label
            with np.load(npz_path) as data:
                support_image = data['image']  # [512, 512]
                support_label = data['label']  # [512, 512]
            support_image = np.stack([support_image]*3, axis=-1)  # 转为3通道 [512,512,3]
            support_image = support_image.astype(np.float32)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1

            support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_npz_path + "\n"))
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot

        # transform
        raw_image = image.copy()
        raw_label = label.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k],
                                                                              support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        return image, label, s_x, s_y, subcls_list, raw_label
        # image：3*512*512， label：512*512， s_x：shot*3*512*512， s_y：shot*512*512， subcls_list：所选类别在self.sub_list中的索引列表

    def __getitem__(self, index):
        if self.data_set == 'Synapse':
            image, label, s_x, s_y, subcls_list, raw_label = self.getitem_synapse(index)
        
         # Return
        if self.mode == 'train':
            # subcls_list 是所选类别在self.sub_list中的索引
            return image, label, s_x, s_y, subcls_list
        elif self.mode == 'val':
            # subcls_list 是所选类别在self.sub_val_list中的索引
            return image, label, s_x, s_y, subcls_list, raw_label


        # # 获取训练image和label
        # image_path, label_path = self.data_list[index] # self.data_list存储了训练集的img和label路径
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        #     raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        # # 获取image包含的所有类别
        # label_class = np.unique(label).tolist()
        # if 0 in label_class: # 去除背景类别
        #     label_class.remove(0)
        # if 255 in label_class:  # 去除忽略类别
        #     label_class.remove(255)
        
        # # 筛选有效类别
        # # 根据运行模式，筛选出当前图像中属于有效类别
        # new_label_class = []
        # for c in label_class: # 遍历image包含的所有类别
        #     # 如果在验证集类别中
        #     if c in self.sub_val_list:  # sub_val_list是验证集类别
        #         if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
        #             new_label_class.append(c)
        #     # 如果在训练集类别中
        #     if c in self.sub_list:  # sub_list是训练集类别
        #         if self.mode == 'train':
        #             new_label_class.append(c)
        # label_class = new_label_class
        # assert len(label_class) > 0

        # # 随机选择一个类别，作为当前任务的目标类别
        # class_chosen = label_class[random.randint(1, len(label_class)) - 1]
        # # 把原始多类标签图，转换为“单类 + 忽略区域”的二值掩码（binary mask）
        # target_pix = np.where(label == class_chosen)  # 找出 label 中 值等于目标类别 (class_chosen) 的所有像素位置
        # ignore_pix = np.where(label == 255) # # 找出 需要忽略的像素（label 值是 255）的位置
        # label[:, :] = 0  # 将整张标签图重置为 0。
        # if target_pix[0].shape[0] > 0:  # 就把目标类别对应的像素标为 1
        #     label[target_pix[0], target_pix[1]] = 1
        # label[ignore_pix[0], ignore_pix[1]] = 255  # 其他都为0

        # # 在sub_class_file_list中，取出对应class_chosen类别的所有图像
        # # file_class_chosen:包含所选类别对应的所有图像
        # file_class_chosen = self.sub_class_file_list[class_chosen]  # sub_class_file_list是验证集的文件
        # num_file = len(file_class_chosen)  # 该类别对应的图像数量

        # # 获得support的 image和label的路径
        # support_image_path_list = []  # 存放选取的support图像 image路径
        # support_label_path_list = []  # 存放选取的support标签 label路径
        # support_idx_list = []  # 存放选取的support图像索引
        # # 随机选择shot个支持集图像
        # for k in range(self.shot):
        #     support_idx = random.randint(1, num_file) - 1
        #     support_image_path = image_path
        #     support_label_path = label_path
        #     # 避免support图像重复，避免support和query是一张图像
        #     while ((support_image_path == image_path and support_label_path == label_path) 
        #         or support_idx in support_idx_list):
        #         support_idx = random.randint(1, num_file) - 1
        #         support_image_path, support_label_path = file_class_chosen[support_idx]
        #     support_idx_list.append(support_idx)
        #     support_image_path_list.append(support_image_path)
        #     support_label_path_list.append(support_label_path)

        # subcls_list = [] # 存储当前选中类别在self.sub_list中的索引
        # if self.mode == 'train':
        #     # 如果 self.sub_list 是 [1, 5, 10, 15] 且 class_chosen 是 10，那么 index() 返回 2。
        #     subcls_list.append(self.sub_list.index(class_chosen))  # self.sub_list是当前fold的训练类别
        # else:
        #     subcls_list.append(self.sub_val_list.index(class_chosen))

        # # 加载support图像，并进行预处理
        # support_image_list_ori = []
        # support_label_list_ori = []
        # support_label_list_ori_mask = []
        # for k in range(self.shot):
        #     # 获取image和label的路径
        #     support_image_path = support_image_path_list[k] 
        #     support_label_path = support_label_path_list[k]
        #     # 读取image和label
        #     support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
        #     support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
        #     support_image = np.float32(support_image)
        #     support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
        #     target_pix = np.where(support_label == class_chosen)
        #     ignore_pix = np.where(support_label == 255)
        #     support_label[:, :] = 0
        #     support_label[target_pix[0], target_pix[1]] = 1

        #     support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
        #     support_label[ignore_pix[0], ignore_pix[1]] = 255
        #     support_label_mask[ignore_pix[0], ignore_pix[1]] = 255
        #     if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
        #         raise (RuntimeError(
        #             "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
        #     support_image_list_ori.append(support_image)
        #     support_label_list_ori.append(support_label)
        #     support_label_list_ori_mask.append(support_label_mask)
        # assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot

        # raw_image = image.copy()
        # raw_label = label.copy()
        # support_image_list = [[] for _ in range(self.shot)]
        # support_label_list = [[] for _ in range(self.shot)]
        # if self.transform is not None:
        #     image, label = self.transform(image, label)
        #     for k in range(self.shot):
        #         support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k],
        #                                                                       support_label_list_ori[k])

        # s_xs = support_image_list
        # s_ys = support_label_list
        # s_x = s_xs[0].unsqueeze(0)
        # for i in range(1, self.shot):
        #     s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        # s_y = s_ys[0].unsqueeze(0)
        # for i in range(1, self.shot):
        #     s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

       
