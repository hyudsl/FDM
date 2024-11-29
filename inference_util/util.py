import os
import cv2
import torch

import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    _image = np.array(image).astype(np.uint8)

    image = _image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    return {"image": image, 'raw_image': _image}

def get_mask(mask_path, im_size):
    mask = Image.open(mask_path)
    if not mask.mode == "RGB":
        mask = mask.convert("RGB")
    mask = np.array(mask).astype(np.float32)
    mask = cv2.resize(mask, im_size[::-1], interpolation=cv2.INTER_NEAREST) # size [w, h]
    mask = 1 - mask / 255.0
    mask = mask[:, :, 0:1]
    mask = np.transpose(mask.astype(np.bool), (2, 0, 1)) # 1 x H x W
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)

    return mask

def load_token(token_paths):
    token_list = []
    for i, path in enumerate(token_paths):
        f = open(path, 'r')
        lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.strip()
            token = line.split()
            tokens = list(map(int, token))
        tokens_tensor = torch.tensor(tokens).view(1,32,32)
        token_list.append(tokens_tensor)
    token_list = torch.cat(token_list, dim=0).cuda()

    return token_list

def get_token_input(data_paths, token_root, iter_num):
    token_paths = []
    for j, path in enumerate(data_paths):
        img_name = os.path.basename(path)
        img_name_wo_ext, _ = os.path.splitext(img_name)
        token_paths.append(os.path.join(token_root, img_name_wo_ext+'_'+str(iter_num)+'.txt'))
        # token_paths.append(os.path.join(token_root, img_name_wo_ext+'_'+'token'+'.txt'))
    token_list = load_token(token_paths)

    return token_list

def tensor2image(input):
    input = input.to(torch.uint8)
    if input.shape[0] == 1:
        input = input.squeeze(0).to('cpu').numpy()
    else:
        input = input.permute(1, 2, 0).to('cpu').numpy()
    input = Image.fromarray(input)

    return input