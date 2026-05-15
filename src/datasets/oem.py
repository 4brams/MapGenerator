import os
import torch.utils.data as data
from PIL import Image
import numpy as np

def oem_cmap(N=256, normalized=False):
    # Extended colormap for all possible pixel values
    cmap = np.random.RandomState(42).randint(0, 256, (N, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]  # background
    if normalized:
        cmap = cmap / 255.0
    return cmap

class OEMSegmentation(data.Dataset):
    cmap = oem_cmap()
    
    def __init__(self, root, image_set='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set
        
        # Handle OEM subfolder if it exists
        oem_path = os.path.join(self.root, 'OEM')
        if os.path.exists(oem_path):
            self.root = oem_path

        split_f = os.path.join(self.root, f'{image_set}.txt')
        if not os.path.exists(split_f):
            raise ValueError(f'Split file {split_f} not found.')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = []
        self.masks = []

        for file_name in file_names:
            region = file_name.rsplit('_', 1)[0]
            base_name = file_name.replace('.tif', '.png')

            image_path = os.path.join(self.root, region, 'images', file_name)
            mask_path = os.path.join(self.root, region, 'masks', base_name)

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.images.append(image_path)
                self.masks.append(mask_path)
            else:
                # Skip missing files with warning
                if not os.path.exists(image_path):
                    print(f'Warning: Missing image {image_path}')
                if not os.path.exists(mask_path):
                    print(f'Warning: Missing mask {mask_path}')

        if len(self.images) == 0:
            raise FileNotFoundError(f'No valid image-mask pairs found in {self.root}')
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')
        
        # Ensure target has same size as image
        if img.size != target.size:
            target = target.resize(img.size, Image.NEAREST)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # Ensure final target is same size as image after transform
        if hasattr(img, 'shape') and hasattr(target, 'shape'):
            # Both are tensors
            if img.shape[-2:] != target.shape[-2:]:
                import torch.nn.functional as F
                target = F.interpolate(target.unsqueeze(0), size=img.shape[-2:], mode='nearest').squeeze(0)
        elif hasattr(img, 'size') and hasattr(target, 'size'):
            # Both are PIL images
            if img.size != target.size:
                target = target.resize(img.size, Image.NEAREST)
        
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]
