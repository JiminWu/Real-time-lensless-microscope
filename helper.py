import argparse, json, math
import scipy.io
import numpy as np
import cv2
import torch
import hdf5storage

import models.fftlayer as fftlayer
from models.unet import Unet


def max_proj(x, axis = 0):
    return np.max(x,axis)

def mean_proj(x, axis = 0):
    return np.mean(x,axis)


def load_saved_args(model_file_path):
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--psf_num', default=5, type=int)
    parser.add_argument('--device', default='0')
    args = parser.parse_args("--device 1".split())

    with open(model_file_path+'args.json', "r") as f:
        args.__dict__=json.load(f)
    return args


def load_model(psf_filepath, model_filepath, device = 'cuda:0', load_model = True):

    args = load_saved_args(model_filepath)
    psfs = hdf5storage.loadmat(psf_filepath)
    psfs = psfs['psfs']
    psfs = crop_psfs(psfs, args.psf_height_org, args.psf_width_org, args.psf_height_crop, args.psf_width_crop, args.psf_shift_height, args.psf_shift_width)

    gamma = np.ones((args.psf_num,1,1))
    fft_stage=fftlayer.fft_conv(psfs, gamma).to(device)
    unet_stage = Unet(n_channel_in=args.psf_num, n_channel_out=1, residual=False, activation='selu').to(device)
    model=fftlayer.MyEnsemble(fft_stage,unet_stage)
    
    if load_model == True:
        model.load_state_dict(torch.load(model_filepath+'model.pt',map_location=torch.device(device)))
        
    return model, args


def crop_psfs(psfs, psf_height_org, psf_width_org, psf_height_crop, psf_width_crop, shift_height, shift_width):
    
    crop_start_x = (psf_height_org - psf_height_crop) // 2 + shift_height
    crop_end_x = psf_height_org - (psf_height_org - psf_height_crop) // 2 + shift_height
    crop_start_y = (psf_width_org - psf_width_crop) // 2 + shift_width
    crop_end_y = psf_width_org - (psf_width_org - psf_width_crop) // 2 + shift_width
    
    psfs = psfs[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    psfs = psfs.transpose(2,0,1)
    
    return psfs

def crop_images(img, image_height_org, image_width_org, image_height_crop, image_width_crop, image_shift_height, image_shift_width):
    
    crop_start_x = (image_height_org - image_height_crop) // 2 + image_shift_height
    crop_end_x = crop_start_x + image_height_crop
    crop_start_y = (image_width_org - image_width_crop) // 2 + image_shift_width
    crop_end_y = crop_start_y + image_width_crop
    
    img = img[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img


def crop_out_single(img, out_height_crop, out_width_crop, image_shift_height, image_shift_width):
    
    h,w = img.shape
    
    crop_start_x = (h - out_height_crop) // 2 - image_shift_height
    crop_end_x = crop_start_x + out_height_crop
    crop_start_y = (w - out_width_crop) // 2 - image_shift_width
    crop_end_y = crop_start_y + out_width_crop
    
    img = img[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img

def crop_out_training(img, out_height_crop, out_width_crop, image_shift_height, image_shift_width):
    
    a,b,h,w = img.shape
    
    crop_start_x = (h - out_height_crop) // 2 - image_shift_height
    crop_end_x = crop_start_x + out_height_crop
    crop_start_y = (w - out_width_crop) // 2 - image_shift_width
    crop_end_y = crop_start_y + out_width_crop
    
    img = img[:,:,crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img
