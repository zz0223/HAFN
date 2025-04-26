import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import gray2rgb

from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_flist, mask_flist, gt_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.input_data = self.load_flist(input_flist)
        self.gt_data = self.load_flist(gt_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.nms = config.NMSMASK_REVERSE

        self.reverse_mask = config.MASK_REVERSE

        print('training:{}    input_data_list:{} gt_data_list:{}'.format(training, input_flist, gt_flist))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.input_data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.input_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load imput_image
        #print("img_index :", index)
        input_img = Image.open(self.input_data[index])
        if input_img.mode != 'RGB':
            input_img = input_img.convert('RGB')
        input_img = np.array(input_img)

        # load gt_image
        gt_img = Image.open(self.gt_data[index])
        if gt_img.mode != 'RGB':
            gt_img = gt_img.convert('RGB')
        gt_img = np.array(gt_img)

        # gray to rgb
        #if len(input_img.shape) < 3:
        #    input_img = gray2rgb(input_img)
        # gray to rgb
        #if len(gt_img.shape) < 3:
        #    gt_img = gray2rgb(gt_img)
        
        # resize/crop if needed
        if size != 0:
            input_img = self.resize(input_img, 512, 352)
        # resize/crop if needed
        if size != 0:
            gt_img = self.resize(gt_img, 512, 352)

        # load mask
        #mask = self.load_mask(input_img, index % len(self.mask_data))
        
        
        
        # augment data
        #if np.random.rand() < 0.5:  # 利用np.random.rand()随机得到一个0-1浮点数，和fliplr比较，判断图片是否左右翻转
        #    input_img = cv2.flip(input_img, 1)
        #    gt_img = cv2.flip(gt_img, 1)
        #    mask = cv2.flip(mask, 1)
        #if np.random.rand() < 0.5:  # 利用np.random.rand()随机得到一个0-1浮点数，和fliplr比较，判断图片是否上下翻转
        #    input_img = cv2.flip(input_img, 0)
        #    gt_img = cv2.flip(gt_img, 0)
        #    mask = cv2.flip(mask, 0)
        
        
        
        #if self.reverse_mask == 1:
            #mask = 255 - mask


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            input_img = input_img[:, ::-1, ...]
            gt_img = gt_img[:, ::-1, ...]

        spec_image = np.uint8(np.clip(np.int32(input_img) - np.int32(gt_img), 0, 255))
        #spec_image = Image.fromarray(spec_image).convert('L')
        #spec_image=np.array(spec_image)
        
        return self.to_tensor(input_img), self.to_tensor(gt_img), self.to_tensor(spec_image)


    def load_mask(self, img, index):
        #print("mask_index :", index)
        imgh, imgw = img.shape[0:2]

        if self.training:
            #mask_index = random.randint(0, len(self.mask_data) - 1)
            mask_index = index
        else:
            mask_index = index
        mask = Image.open(self.mask_data[mask_index])
        channels = mask.mode if mask.mode in ['RGB', 'RGBA'] else 'L'
        if channels in ['RGB', 'RGBA']:
            mask = mask.convert('L')
        mask = np.array(mask)
        mask = self.resize(mask, imgh, imgw)
        #mask = (mask > self.mask_threshold).astype(np.uint8) * 255       # threshold due to interpolation
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255 + (mask <= self.mask_threshold).astype(np.uint8) * 0
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        #if centerCrop and imgh != imgw:
            # center crop
         #   side = np.minimum(imgh, imgw)
         #   j = (imgh - side) // 2
         #   i = (imgw - side) // 2
         #   img = img[j:j + side, i:i + side, ...]
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, dsize=(height, width))

        return img

    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r', encoding='utf-8') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
