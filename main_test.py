# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='models/DnCNN_sigma25', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_001.pth', type=str, help='the model name')

    parser.add_argument('--set_dir', default='1d_data/1d_root', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test results')

    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, img_as_ubyte(np.clip(result, 0, 1)))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    x *= 255
    x = np.asarray(x, dtype=np.uint8)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    args = parse_args()

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    psnrs_ori = []
    ssims_ori = []
    psnrs_pred = []
    ssims_pred = []

    counter = 0
    for im in os.listdir(args.set_dir + '/noisy'):
        # set the counter to only produce 20 test results
        # if counter > 100:
        #     break

        if not im.startswith("test") and im.endswith(".png"):
            x = np.array(imread(os.path.join(args.set_dir, 'noisy', im)), dtype=np.float32) / 255.0
            x_clean = np.array(imread(os.path.join(args.set_dir, 'clean', im)), dtype=np.float32) / 255.0
            # np.random.seed(seed=0)  # for reproducibility
            # y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
            # y = y.astype(np.float32)
            x_torch = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]).cuda()

            torch.cuda.synchronize()
            start_time = time.time()

            noise_torch = model(x_torch)  # inference
            noise = noise_torch.view(x.shape[0], x.shape[1]).cpu().detach().numpy().astype(np.float32)
            x_predict = x - noise

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print('%10s : %2.4f second' % (im, elapsed_time))

            psnr_ori = compare_psnr(x, x_clean, data_range=1.0)
            psnr_pred = compare_psnr(x_predict, x_clean, data_range=1.0)
            ssim_ori = compare_ssim(x, x_clean, multichannel=False)
            ssim_pred = compare_ssim(x_predict, x_clean, multichannel=False)

            if args.save_result:
                name, ext = os.path.splitext(im)
                # show(np.hstack((x, x_predict, x_clean)))  # show the image
                save_result(x_predict, path=os.path.join(args.result_dir, name + '_dncnn' + ext))  # save the denoised image
                with open(os.path.join(args.result_dir, 'results.txt'), 'a') as f:
                    f.write("psnr_ori = {:.3f} psnr_pred: {:.3f} ssim_ori: {:.3f} ssim_pred: {:.3f} \n".format(psnr_ori, psnr_pred, ssim_ori, ssim_pred))

            psnrs_ori.append(psnr_ori)
            ssims_ori.append(ssim_ori)
            psnrs_pred.append(psnr_pred)
            ssims_pred.append(ssim_pred)
            log("psnr_ori = {:.3f} psnr_pred: {:.3f} ssim_ori: {:.3f} ssim_pred: {:.3f}".format(psnr_ori, psnr_pred, ssim_ori, ssim_pred))
            counter += 1

    psnrs_ori = np.array(psnrs_ori)
    log("Avg Test PSNR = {:.3f}dB, Avg Test SSIM = {:.3f} Avg Test PSNR = {:.3f}dB, Avg Test SSIM = {:.3f} ".format(
        np.mean(psnrs_ori[psnrs_ori < 100]), np.mean(ssims_ori), np.mean(psnrs_pred), np.mean(ssims_pred)))







