# Improving Detail in Pluralistic Image Inpainting with Feature Dequantization


## Environment setup
```
git clone https://github.com/hyudsl/FDM.git
cd FDM

conda create -n FDM python==3.8
conda activate FDM

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Prepare dataset


## Training

### Phase 1: Training the base model (PUT)
#### Encoder-decoder
<!-- Training encoder-decoder using the following command. -->
```
d
```
<!-- <details>
  <summary>OR You can make the weights provided by PUT compatible with the model architecture of FDM.</summary>
  ```
  d
  ```
</details> -->

#### Patch-wise feature sampler
<!-- Training patch-wise  -->
```

```

<!-- OR use  -->

## Inference

Set inference configuration with follow structure
```
config_root
    L task1
        L dataloader.yaml
        L model.yaml
        L task_config.yaml
    L task2
    ...
```

Revise data path to your own path in dataloader.yaml
```
dataloader:
  data_root: ./data
  ...
  params:
        name: images/paris/val  # image folder path in data_root
        image_list_file: path/to/image/list # txt file
        ...
        provided_mask_name: masks   # mask folder path in data_root
        provided_mask_list_file: path/to/mask/list # txt file
        ...
```

Run this command to inference
```
# python inference.py --config path/to/config/root --task task/name

# example
python inference.py --config ./configs/paris --task inpainting
```