import torch

from .utils.io import load_image

def get_mae(img1_path, img2_path):
    img1 = load_image(img1_path)['image']
    img2 = load_image(img2_path)['image']
    if torch.is_tensor(img1):
        assert img1.dim() == 4, 'Only batch based data is implemented!'
        mae = (img1.flatten(1)/255.0 - img2.flatten(1)/255.0).abs().sum(dim=-1) / (img1.flatten(1)/255.0 + img2.flatten(1)/255.0).sum(dim=-1)
    else:
        raise NotImplementedError
    return mae.mean()
