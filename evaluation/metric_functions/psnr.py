import math
import torch

import numpy as np

from .utils.io import load_image

def get_PSNR(img1_path, img2_path, tool='none'):
    img1 = load_image(img1_path)['image']
    img2 = load_image(img2_path)['image']
    if tool == 'skimage':
        from skimage.metrics import peak_signal_noise_ratio
        if torch.is_tensor(img1):
            device = img1.device
            assert img1.dim() == 4, "only batch based data is implemented!"
            psnr = []
            img1 = img1.permute(0, 2, 3, 1)/255.0 # B C H W -> B H W C
            img2 = img2.permute(0, 2, 3, 1)/255.0 # B C H W -> B H W C
            for i in range(img1.shape[0]):
                im1_ = img1[i].to('cpu').numpy()
                im2_ = img2[i].to('cpu').numpy()
                if im1_.shape[-1] == 1: # grapy image
                    im1_ = im1_[:,:,0]
                    im2_ = im2_[:,:,0]
                psnr_ = peak_signal_noise_ratio(im1_, im2_)
                psnr.append(psnr_)
            psnr = torch.tensor(psnr, device=device).mean()
        else:
            psnr = peak_signal_noise_ratio(img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0)
    elif tool == 'none':
        PIXEL_MAX = 1
        if torch.is_tensor(img1):
            assert img1.dim() == 4, 'Only batch based data is implemented!'
            mse = ((img1/255.0 - img2/255.0) ** 2).reshape(img1.shape[0], -1).mean(dim=1) # Batch
            mask = mse < 1.0e-10
            mse[mask] = 100.0
            mse[~mask] = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse[~mask]))
            psnr = mse.mean()
        else:
            mse = np.mean( (img1/255. - img2/255.) ** 2 )
            if mse < 1.0e-10:
                return 100
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        raise NotImplementedError
    return psnr
