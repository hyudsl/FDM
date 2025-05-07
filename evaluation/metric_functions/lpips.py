import torch
import random
import lpips
import torchvision.transforms as TF
from PIL import Image

from tqdm import tqdm

from .utils.misc import get_all_file, get_all_subdir

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', 'JPEG'}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, count=None):
        self.files = files
        self.transforms = transforms
        self.count = count

    def __len__(self):
        if self.count is not None:
            return min(self.count, len(self.files))
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert('RGB')
        except:
            raise RuntimeError('File invalid: {}'.format(path))
        if self.transforms is not None:
            img = self.transforms(img)
        img = (img-0.5) / 0.5
        return img

    def get_path(self, index):
        return self.files[index]

def get_dataset(path):
    files = get_all_file(path, end_with=IMAGE_EXTENSIONS)
    random.shuffle(files)
    
    transforms = TF.ToTensor()

    dataset = ImagePathDataset(files, transforms=transforms)
    return dataset

def calculate_lpips_diversity(path, loops, net='vgg'):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    sub_dirs = sorted(get_all_subdir(path, max_depth=2, min_depth=2, path_type='abs'))
    if len(sub_dirs) == 0:
        sub_dirs = sorted(get_all_subdir(path, max_depth=1, min_depth=1, path_type='abs'))
    loss = lpips.LPIPS(net=net, spatial=True).to(device)
    value = []
    for sdi in tqdm(range(len(sub_dirs))):
        sd = sub_dirs[sdi]
        dataset = get_dataset(sd)
    
        processed_pair = set([])
        im_index = list(range(len(dataset)))

        sampled_index = set([])
        with torch.no_grad():
            bar = range(loops)
            if loops > 100:
                bar = tqdm(bar)
            for l in bar:
                if len(im_index) <= 2:
                    continue
                two_idx = tuple(sorted(random.sample(im_index, 2)))
                sampled_index.add(two_idx)
                while two_idx in processed_pair:
                    two_idx = tuple(sorted(random.sample(im_index, 2)))

                processed_pair.add(two_idx)

                im1 = dataset[two_idx[0]].unsqueeze(dim=0).to(device)
                im2 = dataset[two_idx[1]].unsqueeze(dim=0).to(device)
                
                v = loss.forward(im1, im2).detach().mean().to('cpu')
                value.append(v)
        
    if len(value) > 0:
        value = torch.stack(value, dim=0)

    overall_mean = (float(value.sum()/len(value))) if len(value)> 0 else 0

    return overall_mean

def cal_lpips(path1, path2, net='vgg', count=None, im_size=None):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    files1 = sorted(get_all_file(path1, end_with=IMAGE_EXTENSIONS, path_type='abs'))
    files2 = sorted(get_all_file(path2, end_with=IMAGE_EXTENSIONS, path_type='abs'))

    dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
    dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
    loss = lpips.LPIPS(net=net, spatial=True).to(device)
    value = []
    indices = list(range(len(dataset1)))
    for idx in tqdm(indices):
        im1 = dataset1[idx].unsqueeze(dim=0).to(device)
        im2 = dataset2[idx].unsqueeze(dim=0).to(device)

        v = loss.forward(im1, im2).detach().mean().to('cpu')
        value.append(v)

    if len(value) > 0:
        value = torch.stack(value, dim=0)

    overall_mean = (float(value.sum()/len(value))) if len(value)> 0 else 0

    return overall_mean