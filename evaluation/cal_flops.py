import os

import cv2
import mlconfig
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis, flop_count_table

from image_synthesis.modeling.build import build_model
from image_synthesis.utils.io import load_yaml_config

os.environ['CUDA_VISIBLE_DEVICES']="0"

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        batch, generate_config = x

        samples = self.model.sample(batch=inputs, generate_config=None, save_token=False)

        return samples

def load_model(model_config_path, checkpoint_path=None, set_dist=True):
    config = load_yaml_config(model_config_path)
    model = build_model(config)

    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # print(ckpt['model'].keys())

        if 'model' in ckpt:
            # #TODO
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        elif 'state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            missing, unexpected = [], []
            print("====> Warning! No pretrained model!")
        print('Missing keys in created model:\n', missing)
        print('Unexpected keys in state dict:\n', unexpected)

    model.eval()

    model = model.cuda()
    if set_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[None])

    return model

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

def load_mask(mask_path, im_size):
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

config_path = "/home/kr/project/git/FDM/configs/fdm/paris/inpainting"

sample_config_path = os.path.join(config_path, 'task_config.yaml')
sample_config = mlconfig.load(sample_config_path)

model_config_path = os.path.join(config_path, 'model.yaml')
model = load_model(model_config_path, set_dist=False)

model = ModelWrapper(model)

image = load_image('/home/kr/project/data/paris/resized_val/0.png')['image']
mask = load_mask('/home/kr/project/data/testing_mask_dataset/02000.png', image.shape[2:])
inputs = {'image': image, 'mask': mask}

flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops))

# print(flops.by_operator())