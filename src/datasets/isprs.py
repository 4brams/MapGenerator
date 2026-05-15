import os
import re
import torch.utils.data as data
from PIL import Image
import numpy as np


def isprs_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap / 255
    return cmap


class ISPRSSegmentation(data.Dataset):
    """ISPRS Semantic Segmentation Dataset.

    The dataset root should contain the following folders:
        - Images/
        - Labels/

    Label colors expected in Labels/*.tif are:
        (0, 0, 255), (0, 255, 0), (0, 255, 255),
        (255, 0, 0), (255, 255, 0), (255, 255, 255)

    Optionally, for split support, add either:
        - splits/train.txt and splits/val.txt
      or
        - Images/train/, Images/val/ and optional Labels/train/, Labels/val/

    Each split file may contain image basenames or filenames.
    """

    label_colors = np.array([
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 0],
        [255, 255, 0],
        [255, 255, 255],
    ], dtype=np.uint8)
    num_classes = len(label_colors)
    cmap = label_colors

    def __init__(self, root, image_set='train', transform=None):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'Images')
        self.labels_dir = os.path.join(self.root, 'Labels')
        self.split_dir = os.path.join(self.root, 'splits')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
            alt_root = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'rawDataset', 'ISPRS'))
            if os.path.isdir(alt_root):
                self.root = alt_root
                self.images_dir = os.path.join(self.root, 'Images')
                self.labels_dir = os.path.join(self.root, 'Labels')
                self.split_dir = os.path.join(self.root, 'splits')

        self._check_directories()
        self.images = self._load_image_paths()
        self.label_map = self._build_label_map()
        self.masks = [self._find_mask_path(img_path) for img_path in self.images]

        if len(self.images) == 0:
            raise RuntimeError('No images found for ISPRS dataset at %s' % self.images_dir)
        if len(self.images) != len(self.masks):
            raise RuntimeError('Number of images and masks do not match for ISPRS dataset.')

    def _check_directories(self):
        if not os.path.isdir(self.images_dir):
            raise RuntimeError('ISPRS images folder not found: %s' % self.images_dir)
        if not os.path.isdir(self.labels_dir):
            raise RuntimeError('ISPRS labels folder not found: %s' % self.labels_dir)

    def _load_image_paths(self):
        images = []
        file_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        split_file = os.path.join(self.split_dir, self.image_set + '.txt')

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f if line.strip()]
            all_images = self._list_files(self.images_dir, file_extensions)
            image_map = {os.path.splitext(os.path.basename(path))[0]: path for path in all_images}
            for name in file_names:
                if os.path.splitext(name)[1]:
                    path = os.path.join(self.images_dir, name)
                    if os.path.exists(path):
                        images.append(path)
                        continue
                base = os.path.splitext(os.path.basename(name))[0]
                if base in image_map:
                    images.append(image_map[base])
                    continue
                candidates = [path for key, path in image_map.items() if key.endswith(base)]
                if len(candidates) == 1:
                    images.append(candidates[0])
                    continue
                raise RuntimeError('Cannot resolve image name "%s" from split file %s' % (name, split_file))
            return images

        image_subdir = os.path.join(self.images_dir, self.image_set)
        label_subdir = os.path.join(self.labels_dir, self.image_set)
        if os.path.isdir(image_subdir):
            if os.path.isdir(label_subdir):
                self.labels_dir = label_subdir
            return self._list_files(image_subdir, file_extensions)

        if self.image_set.lower() in ['train', 'val']:
            all_images = self._list_files(self.images_dir, file_extensions)
            if len(all_images) == 0:
                raise RuntimeError('No ISPRS images found in %s' % self.images_dir)
            split_index = int(len(all_images) * 0.8)
            if self.image_set.lower() == 'train':
                return all_images[:split_index]
            return all_images[split_index:]

        if self.image_set.lower() in ['trainval', 'test']:
            raise ValueError(
                'ISPRS split "%s" not found. Please add split file in %s or use '
                'Images/train, Images/val and Labels/train, Labels/val folders.' % (
                    self.image_set, self.split_dir)
            )

        return self._list_files(self.images_dir, file_extensions)

    def _list_files(self, root, suffixes):
        files = []
        for entry in sorted(os.listdir(root)):
            if entry.lower().endswith(suffixes):
                files.append(os.path.join(root, entry))
        return files

    def _build_label_map(self):
        file_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        label_paths = self._list_files(self.labels_dir, file_extensions)
        label_map = {os.path.splitext(os.path.basename(path))[0]: path for path in label_paths}
        return label_map

    def _find_mask_path(self, img_path):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [
            image_name,
            image_name.replace('Image_', 'Label_'),
            image_name.replace('image_', 'label_'),
        ]

        numeric = re.search(r'(\d+)$', image_name)
        if numeric:
            candidates.append('Label_' + numeric.group(1))
            candidates.append('label_' + numeric.group(1))

        for key in candidates:
            if key in self.label_map:
                return self.label_map[key]

        raise RuntimeError('No label found for image "%s". Checked candidates: %s' %
                           (img_path, candidates))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)
        target = self.encode_target(target)
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, target):
        target = np.array(target, dtype=np.uint8)
        if target.ndim == 3:
            label = np.full(target.shape[:2], 255, dtype=np.uint8)
            for idx, color in enumerate(cls.label_colors):
                mask = np.all(target == color, axis=-1)
                label[mask] = idx
            return label
        label = np.full(target.shape, 255, dtype=np.uint8)
        return label

    @classmethod
    def decode_target(cls, mask):
        target = np.array(mask, dtype=np.int32)
        ignore_mask = target == 255
        target[ignore_mask] = 0
        decoded = cls.cmap[target]
        if np.any(ignore_mask):
            decoded[ignore_mask] = np.array([0, 0, 0], dtype=np.uint8)
        return decoded
