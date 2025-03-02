import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF


def load_image(filename):
    # ext = splitext(filename)[1]
    # if ext == '.npy':
    #     return Image.fromarray(np.load(filename))
    # elif ext in ['.pt', '.pth']:
    #     return Image.fromarray(torch.load(filename).numpy())
    # else:
    return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        if isinstance(scale, tuple):
            assert (0 < s <= 1 for s in scale), 'Scale must be between 0 and 1'
        else:
            assert (0 < scale <=1), 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids2 = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        #Check one
        #print(len(self.ids))

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids2),
                total=len(self.ids2)
            ))

        #self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        self.mask_values = [0, 1]
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)


        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        name_mask = self.ids2[idx]
        mask_file = list(self.mask_dir.glob(name_mask + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        self.img_size = 224
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        #assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        img = load_image(img_file[0])
        #if len(mask_file) == 0: print("mask file Not loading")
        mask = load_image(mask_file[0])

        # img = np.array(self.resize(img, resize_to=224), np.float32)
        # mask = np.array(self.resize(mask, resize_to=224), np.uint8)
        mask = TF.resize(mask, img.size, interpolation=Image.BICUBIC)

        mask = mask.transpose(Image.TRANSPOSE)
        # print(img.size, mask.size)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

        # def resize(self, img, resize_to):
        #     w, h = img.size
        #     scale = max(w, h) / resize_to
        #     if w > h:
        #         h = int(h // scale)
        #         img = img.resize((224, h))
        #     elif w == h:
        #         img = img.resize((224, 224))
        #     else:
        #         w = int(w // scale)
        #
        #         img = img.resize((w, 224))
        #     return img


class CBIS_DDSM(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')

