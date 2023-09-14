import torch

from torch.autograd import Variable
#from evaluate import test_main 

import numpy as np

import cv2
from scipy import ndimage
from torchvision import transforms

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0
    return im_ycbcr

def makeSpatialActivityMap(im):
  im = im.cpu().detach().permute(0, 2, 3, 1).numpy()[0]
  #H = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8  
  im = rgb2ycbcr(im)
  im_sob = ndimage.sobel(im[:,:,0])
  im_zero = np.zeros_like(im_sob)
  im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]

  maxval = im_zero.max()

  if maxval == 0:
    im_zero = im_zero + 1
    maxval = 1
  
  im_sob = im_zero /maxval

  DF = np.array([[0, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 0]]).astype('uint8')
  
  out_im = cv2.dilate(im_sob, DF)
  return out_im
          

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    iters = 10
    lr = 0.005

    sp_map = makeSpatialActivityMap(compress_image * 255)
    sp_map = sp_map / 255
    sp_map = transforms.ToTensor()(sp_map.astype(np.float32))
    sp_map = sp_map.unsqueeze_(0)
    sp_map = sp_map.to(device)

    
    compress_image = Variable(compress_image, requires_grad=True)
    opt = torch.optim.Adam([compress_image], lr = lr)
    
    for i in range(iters):
        score = model(ref_image.to(device), compress_image.to(device)) if ref_image is not None else model(compress_image.to(device))
        loss = 1 - score / metric_range
        loss.backward() 
        compress_image.grad *= sp_map
        opt.step()
        compress_image.data.clamp_(0., 1.)
        opt.zero_grad()

    res_image = (compress_image).data.clamp_(min=0, max=1)

    return res_image

# if __name__ == "__main__":
#     test_main(attack)

