import os
import cv2
import tqdm
import torch
import warnings
import mlconfig
import argparse

import numpy as np

from PIL import Image

import torch.distributed as dist

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.distributed.launch import launch
from image_synthesis.modeling.build import build_model

from inference_util.util import get_token_input, load_token, load_image, get_mask, tensor2image

NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

def load_model(model_config_path, checkpoint_path=None, set_dist=True):
    config = load_yaml_config(model_config_path)
    model = build_model(config)

    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

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

def get_dataloader(dataloader_config, set_dist):
    dataset_cfg = dataloader_config['dataloader']
    
    val_dataset = []
    for ds_cfg in dataset_cfg['validation_datasets']:
        ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')
        ds = instantiate_from_config(ds_cfg)
        val_dataset.append(ds)
    if len(val_dataset) > 1:
        val_dataset = ConcatDataset(val_dataset)
    else:
        val_dataset = val_dataset[0]

    if set_dist:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, seed=123)
        val_iters = len(val_sampler) // dataset_cfg['batch_size']

        num_workers = dataset_cfg['num_workers']

        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=dataset_cfg['batch_size'], 
                                                shuffle=False, #(val_sampler is None),
                                                num_workers=num_workers, 
                                                pin_memory=True, 
                                                sampler=val_sampler, 
                                                drop_last=False)
    else:
        val_iters = len(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=False)

    return {'dataloader': val_loader, 'iterations': val_iters}

def inference(model, dataloader_info, sample_config, set_dist):
    dataloader = dataloader_info['dataloader']
    iterations = dataloader_info['iterations']

    save_root = sample_config.save_path

    sample_num = sample_config.sample_num
    save_token = True if not sample_config.generate_with_token else False

    if not set_dist or dist.get_rank() == 0:
        tq = tqdm.tqdm(total=iterations)
    for i, data in enumerate(dataloader):
        paths = data['path']
        inputs = {'image': data['image'], 'mask': data['mask']}

        for l in range(sample_num):
            if sample_config.generate_with_token:
                inputs['token'] = get_token_input(data['path'], sample_config.token_root, l)

            save_path = os.path.join(save_root, 'inpainted_image')
            img_name = os.path.basename(paths[0])
            img_name_wo_ext, ext = os.path.splitext(img_name)
            check_file_path = os.path.join(save_path, img_name_wo_ext+"_"+str(l)+ext)
            if os.path.isfile(check_file_path):
                continue

            if set_dist:
                outputs = model.module.sample(batch=inputs, generate_config=sample_config.generate_config, save_token=save_token)
            else:
                outputs = model.sample(batch=inputs, generate_config=sample_config.generate_config, save_token=save_token)

            for output_name in outputs:
                save_path = os.path.join(save_root, output_name)
                os.makedirs(save_path, exist_ok=True)

                for j, output in enumerate(outputs[output_name]):
                    img_name = os.path.basename(paths[j])
                    img_name_wo_ext, ext = os.path.splitext(img_name)

                    if output_name == 'token':
                        token_path = os.path.join(save_path, img_name_wo_ext+'_'+str(l)+'.txt')
                        f = open(token_path, 'w')

                        data = ''
                        for t in output:
                            data += str(t.item())+' '
                        data += '\n'
                        f.write(data)
                        f.close()

                    elif output_name == 'inpainted_image':
                        outptut = tensor2image(output)
                        outptut.save(os.path.join(save_path, img_name_wo_ext+"_"+str(l)+ext))
                    
                    else:
                        outptut = tensor2image(output)
                        outptut.save(os.path.join(save_path, img_name))

        if not set_dist or dist.get_rank() == 0:
            tq.set_postfix(name=img_name)
            tq.update(1)
    if not set_dist or dist.get_rank() == 0:
        tq.close()

def main(local_rank, args):
    config_path = args.config
    task = args.task
    config_path = os.path.join(config_path, task)

    sample_config_path = os.path.join(config_path, 'task_config.yaml')
    model_config_path = os.path.join(config_path, 'model.yaml')
    dataloader_config_path = os.path.join(config_path, 'dataloader.yaml')

    sample_config = mlconfig.load(sample_config_path)

    model = load_model(model_config_path, set_dist=args.distributed)

    dataloader_config = load_yaml_config(dataloader_config_path)
    dataloader_info = get_dataloader(dataloader_config, set_dist=args.distributed)

    inference(model, dataloader_info, sample_config, set_dist=args.distributed)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    # args for ddp
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'mpi', 'gloo'],
                        help='which type of bakend for ddp')
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL, 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    parser.add_argument('--config', type=str, default='./configs/paris',
                        help='')
    parser.add_argument('--task', type=str, default='inpainting',
                        help='')

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )  

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args

if __name__ == '__main__':

    args = get_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    args.distributed = args.world_size > 1

    launch(main, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args.backend, args=(args,))