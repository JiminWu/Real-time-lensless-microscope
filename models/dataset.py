import skimage.io
#import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch
import numpy as np
import scipy.io


class load_data(Dataset):

    def __init__(self, all_files, filepath_meas, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width):#,transform=None):

        self.all_files_gt =  all_files
        self.filepath_meas = filepath_meas
        self.image_height_org = image_height_org
        self.image_width_org = image_width_org
        self.image_height_crop = image_height_crop
        self.image_width_crop = image_width_crop
        self.image_shift_height = image_shift_height
        self.image_shift_width = image_shift_width
        

    def __len__(self):
        return len(self.all_files_gt)

    def __getitem__(self, idx):

        im_gt = skimage.io.imread(self.all_files_gt[idx])
        im_meas = skimage.io.imread(self.filepath_meas[idx])
        sample = {'im_gt': im_gt.astype('float32')/255., 'meas': im_meas.astype('float32')/255.}
        
        im_gt, meas = sample['im_gt'], sample['meas']
        
        crop_start_x = (self.image_height_org - self.image_height_crop) // 2 + self.image_shift_height
        crop_end_x = self.image_height_org - (self.image_height_org - self.image_height_crop) // 2 + self.image_shift_height
        crop_start_y = (self.image_width_org - self.image_width_crop) // 2 + self.image_shift_width
        crop_end_y = self.image_width_org - (self.image_width_org - self.image_width_crop) // 2 + self.image_shift_width
        
        meas= meas[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
        im_gt= im_gt[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
        
        return {'im_gt': torch.from_numpy(im_gt).unsqueeze(0),
                'meas': torch.from_numpy(meas).unsqueeze(0)}
