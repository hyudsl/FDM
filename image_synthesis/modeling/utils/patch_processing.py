import torch.nn.functional as F

def img2patches(data, patch_size, patch_num):
    B, C, H, W = data.shape

    if isinstance(patch_num, list):
        h, w = patch_num
    else:
        h, w = patch_num, patch_num

    patches = F.unfold(data, kernel_size=[patch_size,patch_size], stride=[patch_size,patch_size])   # Bx(C*PS^2)x(PN^2)
    patches = patches.permute(0,2,1)                                                                # Bx(PN^2)x(C*PS^2)
    patches = patches.reshape(B*h*w,C,patch_size,patch_size)                        # (B*PN^2)xCxPSxPS

    return patches

def patches2img(data, out_size, patch_size, patch_num):
    if isinstance(out_size, list):
        H, W = out_size
    else:
        H, W = out_size, out_size

    if isinstance(patch_num, list):
        h, w = patch_num
    else:
        h, w = patch_num, patch_num

    B = int(data.shape[0]/(h*w))
    img = data.reshape(B, h*w, -1)
    img = img.permute(0,2,1)
    img = F.fold(img, output_size=[H, W], kernel_size=[patch_size,patch_size], stride=[patch_size,patch_size])
    return img
