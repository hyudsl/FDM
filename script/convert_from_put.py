import os
import torch
import mlconfig

from image_synthesis.utils.io import load_yaml_config

from image_synthesis.modeling.build import build_model

def load_model(model_config_path, checkpoint_path=None):
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

    return model

if __name__ == '__main__':
    config_path = './configs/bte_lip/ffhq/bte'
    task = 'single'
    config_path = os.path.join(config_path, task)

    sample_config_path = os.path.join(config_path, 'task_config.yaml')
    sample_config = mlconfig.load(sample_config_path)
    model_config_path = os.path.join(config_path, 'model.yaml')
    checkpoint_path = sample_config.checkpoint_path

    save_path = os.path.dirname(checkpoint_path)
    sample_config.save_path = save_path

    model = load_model(model_config_path, checkpoint_path)
    # model = load_model(model_config_path)
    model.cuda()

    model.content_codec.encoder.convert()
    model.content_codec.decoder.convert()

    state_dict = {
        'model': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict() 
    }
    
    torch.save(state_dict, os.path.join(save_path,"converted.pth"))
