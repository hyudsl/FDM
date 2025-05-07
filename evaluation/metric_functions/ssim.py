import cv2
import torch

import numpy as np

from .utils.io import load_image

def get_SSIM(img1_path, img2_path, full=True, win_size=None):
    img1 = load_image(img1_path)['image']
    img2 = load_image(img2_path)['image']
    from skimage.metrics import structural_similarity
    def get_one_pair_image(im1, im2):
        if len(im1.shape) == 3:
            if im1.shape[-1] == 3:
                im1_ = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                im1_ = im1[:, :, 0].astype(np.uint8)
        else:
            im1_ = im1.copy().astype(np.uint8)
        if len(im2.shape) == 3:
            if im2.shape[-1] == 3:
                im2_ = cv2.cvtColor(im2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                im2_ = im2[:, :, 0].astype(np.uint8)
        else:
            im2_ = im2.copy().astype(np.uint8)
        
        # from skimage.measure import compare_ssim
        if full:
            score, diff = structural_similarity(im1_, im2_, full=full, win_size=win_size)
            diff = (diff * 255).astype("uint8")
        else:
            score = structural_similarity(im1_, im2_, full=full, win_size=win_size)
        return score
    
    if torch.is_tensor(img1): # tensor
        is_tensor = True
        device = img1.device
        # import pdb; pdb.set_trace()
        if len(img1.shape) == 3: # H, W ,C
            img1 = img1.unsqueeze(dim=0).to('cpu').numpy() # 1 H W C
            img2 = img2.unsqueeze(dim=0).to('cpu').numpy()
        else: # B C H W
            img1 = img1.permute(0, 2, 3, 1).to('cpu').numpy() # B H W C
            img2 = img2.permute(0, 2, 3, 1).to('cpu').numpy()
    else:
        is_tensor = False
        img1 = img1.copy()[np.newaxis, ...]
        img2 = img2.copy()[np.newaxis, ...]
    
    score = 0
    count = img1.shape[0]
    for i in range(count):
        im1 = img1[i]
        im2 = img2[i]
        score += get_one_pair_image(im1, im2)
    if is_tensor:
        score = torch.tensor(score).to(device)
    return score / count
