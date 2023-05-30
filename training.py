import numpy as np
import torch, torch.optim
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import os, sys, json, glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from math import ceil
import argparse
from PIL import Image
import hdf5storage

from torch.utils.data import Dataset, DataLoader
import cv2

import models.fftlayer as fftlayer
import models.dataset as ds
import models.discriminator as dis
import helper as hp
import models.loss as L
from models.unet import Unet


parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--device', default='0')
parser.add_argument('--load_path',default=None)
#parser.add_argument('--load_path',default='model.pt')#default=None
parser.add_argument('--save_checkponts',default=True)

# Loss functions
parser.add_argument('--lambda_SSIM',default=1.0)
parser.add_argument('--lambda_L1',default=1.0)
parser.add_argument('--lambda_adversarial',default=0.03)

# PSF parameters
parser.add_argument('--psf_num', default=5, type=int)
parser.add_argument('--psf_height_org',default=2048)
parser.add_argument('--psf_width_org',default=2048)
parser.add_argument('--psf_height_crop',default=1536)
parser.add_argument('--psf_width_crop',default=1536)
parser.add_argument('--psf_shift_height',default=-56)
parser.add_argument('--psf_shift_width',default=-56)

device = 'cuda:0'
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# Load PSFs (matlab file)
registered_psfs_path = './sample_data/psf5_2048.mat'
psfs = hdf5storage.loadmat(registered_psfs_path)
psfs=psfs['psfs']
psfs=hp.crop_psfs(psfs, args.psf_height_org, args.psf_width_org, args.psf_height_crop, args.psf_width_crop, args.psf_shift_height, args.psf_shift_width)
gamma = np.ones((args.psf_num,1,1))

# Image cropping parameters
image_height_org = 2048
image_width_org = 2048
image_height_crop = 1536
image_width_crop = 1536
image_shift_height = -56
image_shift_width = -56

# Target reconstruction parameters
target_height = 768
target_width = 768

# Training/testing dataset path
filepath_gt = 'E:/data/DataSet_1.53_0930_new/processed_gt_2048/' #processed_gt_2048/'
filepath_meas = 'E:/data/DataSet_1.53_0930_new/simulated_capture_2048/' #processed_cap_2048/'
filepath_val_gt = 'E:/data/DataSet_1.53_0930_new/processed_val_gt/' #processed_val_gt/'
filepath_val_meas = 'E:/data/DataSet_1.53_0930_new/val_capture/' #processed_val_cap/'

filepath_gt= glob.glob(filepath_gt+'*')
filepath_meas= glob.glob(filepath_meas+'*')
filepath_val_gt= glob.glob(filepath_val_gt+'*')
filepath_val_meas= glob.glob(filepath_val_meas+'*')

print('training images:', len(filepath_gt),
          'testing images:', len(filepath_val_gt))

dataset_train = ds.load_data(filepath_gt, filepath_meas, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)

dataset_test = ds.load_data(filepath_val_gt, filepath_val_meas, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

unet_layer = Unet(n_channel_in=args.psf_num, n_channel_out=1, residual=False, activation='selu').to(device)
fft_layer = fftlayer.fft_conv(psfs, gamma).to(device)
model = fftlayer.MyEnsemble(fft_layer,unet_layer)

D = dis.Discriminator(args).to(device)
D_model = dis.DEnsemble(D)
Dloss = L.DLoss(args).to(device)
Gloss = L.GLoss(args).to(device)

if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path, map_location=torch.device(device)))
    print('loading saved model')


if __name__ == '__main__':

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    Doptimizer = torch.optim.Adam(D_model.parameters(), lr = args.lr)
    
    if args.save_checkponts == True:
        filepath_save = 'saved_data/' +"trained_test0410/"
    
        if not os.path.exists(filepath_save):
            os.makedirs(filepath_save)
    
        with open(filepath_save + 'args.json', 'w') as fp:
            json.dump(vars(args), fp)
    
    best_loss=1e6
    
    D_ds = 0.75
    D_new_size = (ceil(D_ds*target_height), ceil(D_ds*target_width))
    
    for itr in range(0,args.epochs):
        for i_batch, sample_batched in enumerate(dataloader_train):
            optimizer.zero_grad()
            Doptimizer.zero_grad()
            
            out = model(sample_batched['meas'].to(device))
            out = hp.crop_out_training(out, target_height, target_width, image_shift_height, image_shift_width)
            gt = sample_batched['im_gt'].to(device)
            gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)

            downsized_out = F.interpolate(out, size=D_new_size, mode='bilinear', align_corners=False)
            downsized_gt = F.interpolate(gt, size=D_new_size, mode='bilinear', align_corners=False)
            real_logit = D(downsized_gt)
            fake_logit = D(downsized_out)
    
            Dloss(real_logit=real_logit, fake_logit=fake_logit)
            Dloss.total_loss.backward()
            Doptimizer.step()
            
            out = model(sample_batched['meas'].to(device))
            out = hp.crop_out_training(out, target_height, target_width, image_shift_height, image_shift_width)
            gt = sample_batched['im_gt'].to(device)
            gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)

            downsized_out = F.interpolate(out, size=D_new_size, mode='bilinear', align_corners=False)
            fake_logit = D(downsized_out)
            
            Gloss(output=out, target=gt,fake_logit=fake_logit)
            Gloss.total_loss.backward()
            optimizer.step()
            
            print('epoch: ', itr, ' batch: ', i_batch, ' loss: ', Gloss.total_loss.item(), end='\r')
  
        output_numpy = out.detach().cpu().numpy()[0][0]
        gt_numpy = gt.detach().cpu().numpy()[0][0]
        meas_numpy = sample_batched['meas'].detach().cpu().numpy()[0][0]
        
        if args.save_checkponts == True:
            torch.save(model.state_dict(), filepath_save + 'model_noval.pt')
            torch.save(D_model.state_dict(), filepath_save + 'D_model_noval.pt')
        
        
        if itr%1==0:
            total_loss=0
            for i_batch, sample_batched in enumerate(dataloader_test):
                with torch.no_grad():
                    out = model(sample_batched['meas'].to(device))
                    out = hp.crop_out_training(out, target_height, target_width, image_shift_height, image_shift_width)
                    gt = sample_batched['im_gt'].to(device)
                    gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)
        
                    downsized_out = F.interpolate(out, size=D_new_size, mode='bilinear', align_corners=False)
                    fake_logit = D(downsized_out)
                    
                    Gloss(output=out, target=gt,fake_logit=fake_logit)
                    total_loss+=Gloss.total_loss.item()
                    
                    print('loss for testing image ',itr,' ',i_batch, Gloss.total_loss.item())

            print('Total loss for testing set ',itr,' ', total_loss)
                 
                 
            if args.save_checkponts == True:
                im_gt = Image.fromarray((np.clip(gt_numpy/np.max(gt_numpy),0,1)*255).astype(np.uint8))
                im = Image.fromarray((np.clip(output_numpy/np.max(output_numpy),0,1)*255).astype(np.uint8))
                im.save(filepath_save + str(itr) + '.png')
                im_gt.save(filepath_save + str(itr) + 'gt.png')
            
            
            if total_loss<best_loss:
                best_loss=total_loss
    
                # save checkpoint
                if args.save_checkponts == True:
                    torch.save(model.state_dict(), filepath_save + 'model.pt')
                    torch.save(D_model.state_dict(), filepath_save + 'D_model.pt')

        
