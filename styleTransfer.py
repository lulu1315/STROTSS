import time
import math
import sys

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imageio import imread, imwrite

from   st_helper import *
import utils
from   utils import *

#def run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance,regions, output_path='./output.png'):
def run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance,regions, output_path, target_size, weight_decay,max_iters,loss_treshold):

    #smll_sz = 64
    for scl in range(1,max_scl-1):
        target_size=target_size/2
        
    #print ('target_size : ',target_size)
    smll_sz = int(target_size)
    #print ('smll_sz : ',smll_sz)
    start = time.time()
    total_iters=0
    scale_iters=0
    
    min_lr=1e-3
    max_lr=2e-3
    
    #learning rate as a linear equation
    m=(min_lr-max_lr)/(max_scl-2)
    p=max_lr-m
    
    content_im_big = utils.to_device(Variable(load_path_for_pytorch(content_path,target_size,force_scale=True).unsqueeze(0)))

    for scl in range(1,max_scl):

        long_side = smll_sz*(2**(scl-1))
        #lr = 2e-3
        lr = m*scl+p
        
        print('----------')
        print('[styleTransfer] scl: ' , scl)
        print('[styleTransfer] long_side: ' , long_side)
        print('[styleTransfer] weight_decay: ' , weight_decay)
        print('[styleTransfer] content_weight: ' , content_weight)        
        
        ### Load Style and Content Image ###
        content_im = utils.to_device(Variable(load_path_for_pytorch(content_path,long_side,force_scale=True).unsqueeze(0)))
        content_im_mean = utils.to_device(Variable(load_path_for_pytorch(style_path,long_side,force_scale=True).unsqueeze(0))).mean(2,keepdim=True).mean(3,keepdim=True)
        
        ### Compute bottom level of laplaccian pyramid for content image at current scale ###
        lap = content_im.clone()-F.interpolate(F.interpolate(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear',align_corners=True),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)
        nz = torch.normal(lap*0.,0.1)


        canvas = F.interpolate(torch.clamp(lap,-0.5,0.5),(content_im_big.size(2),content_im_big.size(3)),mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)

        if scl == 1:
            canvas = F.interpolate(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)

        ### Initialize by zeroing out all but highest and lowest levels of Laplaccian Pyramid ###
        if scl == 1:
            if 1:
                stylized_im = Variable(content_im_mean+lap)
            else:
                stylized_im = Variable(content_im.data)

        ### Otherwise bilinearly upsample previous scales output and add back bottom level of Laplaccian pyramid for current scale of content image ###
        if scl > 1 and scl < max_scl-1:
            stylized_im = F.interpolate(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)+lap

        #if scl > 3:
        if scl > max_scl-2:
            stylized_im = F.interpolate(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)
            #lr = 1e-3
        
        print('[styleTransfer] learning rate: ' , lr)
        print('----------')
        ### Style Transfer at this scale ###
        stylized_im, final_loss, scale_iters = style_transfer(stylized_im,content_im,style_path,output_path,scl,max_scl,max_iters,loss_treshold,long_side,0.,use_guidance=use_guidance,coords=coords, content_weight=content_weight,lr=lr,regions=regions)
        total_iters=total_iters+scale_iters
        
        canvas = F.interpolate(torch.clamp(stylized_im,-0.5,0.5),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)
        
        ### Decrease Content Weight for next scale ###
        content_weight = content_weight/weight_decay

    print("Finished in: ", int(time.time()-start), 'Seconds')
    print('Final Loss: ', final_loss)
    print('Total number of iterations: ', total_iters)

    canvas = torch.clamp(stylized_im[0],-0.5,0.5).data.cpu().numpy().transpose(1,2,0)
    imwrite(output_path,canvas)
    return final_loss , stylized_im

if __name__=='__main__':

    ### Parse Command Line Arguments ###
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    output_path = sys.argv[3]
    max_scl = int(sys.argv[5])+1
    weight_decay=float(sys.argv[7])
    #content_weight = float(sys.argv[4])*16.0
    #content_weight = float(sys.argv[4])*(2**(max_scl-2))
    content_weight = float(sys.argv[4])*(weight_decay**(max_scl-2))
    target_size=int(sys.argv[6])
    max_iters=int(sys.argv[8])
    loss_treshold=float(sys.argv[9])
    
    use_guidance_region = '-gr' in sys.argv
    use_guidance_points = False
    use_gpu = not ('-cpu' in sys.argv)
    utils.use_gpu = use_gpu


    paths = glob(style_path+'*')
    losses = []
    ims = []


    ### Preprocess User Guidance if Required ###
    coords=0.
    if use_guidance_region:
        i = sys.argv.index('-gr')
        regions = utils.extract_regions(sys.argv[i+1],sys.argv[i+2])
    else:
        try:
            regions = [[imread(content_path)[:,:,0]*0.+1.], [imread(style_path)[:,:,0]*0.+1.]]
        except:
            regions = [[imread(content_path)[:,:]*0.+1.], [imread(style_path)[:,:]*0.+1.]]

    ### Style Transfer and save output ###
    loss,canvas = run_st(content_path,style_path,content_weight,max_scl,coords,use_guidance_points,regions,output_path,target_size,weight_decay,max_iters,loss_treshold)
