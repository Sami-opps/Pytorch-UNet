from random import shuffle
import torch 
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import os 
import numpy as np 
import math 
from typing import Tuple, List, Union 


class Dataset(Dataset):
    def __init__(self, root, 
                 img_size=224, 
                 classes=1,     # Binary. Yours.  Without Background
                 mode='train', 
                 train_val_split=0.8, 
                 transforms=None) -> None:
        super().__init__()
        
        self.img_size = img_size 
        self.classes = classes 
        self.transforms = transforms
        
        if isinstance(img_size, (Tuple, List)):
            assert img_size[0] == img_size[1], 'Only supports equal size of image' 
            self.img_size = img_size[0]
        
        imgs_root = os.path.join(root, 'imgs')
        masks_root = os.path.join(root, 'masks')
        img_files = os.listdir(imgs_root)
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        
        # for img in img_files: mask_1_1-263_copy1 mask_1_2-241_copy2
        #     if os.path.exists(imgs_root):
        #         self.img_paths.append(os.path.join(imgs_root, img))

        for img in img_files:
            mask_name = img[:-3] + '.jpg'
            mask_path = os.path.join(masks_root, mask_name)
            if os.path.exists(mask_path):
                self.img_paths.append(os.path.join(imgs_root, img))
                self.mask_paths.append(mask_path)
            else:
                not_found.append(img)
                
        if len(not_found) > 0:
            print(f'{len(not_found)} images have no masks. Training on found masks')
            
        if len(self.img_paths) < 1:
            raise "No images have a matching mask. Make sure the images have corresponding mask"

        # split Training and valid 
        train_size = int(len(self.img_paths)*train_val_split)
        
        if mode in ['train', 'training']:
            self.img_paths = self.img_paths[:train_size]
            self.mask_paths = self.mask_paths[:train_size]
        else:
            self.img_paths = self.img_paths[train_size:]
            self.mask_paths = self.mask_paths[train_size:]
            
            
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        if self.classes == 1:
            mask = Image.open(self.mask_paths[idx]).convert('1')
        else:
            mask = Image.open(self.mask_paths[idx])
        
        image = np.array(self.resize(image, resize_to=self.img_size), np.float32)
        mask = np.array(self.resize(mask, resize_to=self.img_size), np.uint8)
        
        if len(image.shape) == 2: # Gray or Binary
            image = image[:, :, None] # add channel. 
            w, h, c = image.shape 
        elif len(image.shape) == 3: # RGB 
            w, h, c = image.shape
            
        if w < h:
            r, l = math.floor((self.img_size-w)/2), math.ceil((self.img_size-w)/2)
            image = np.pad(image, ((r, l), (0, 0), (0, 0)))
            mask = np.pad(mask, ((r, l), (0, 0) ))
        elif w > h:
            r, l = math.floor((self.img_size-h)/2), math.ceil((self.img_size-h)/2)
            image = np.pad(image, ((0, 0), (r, l), (0, 0)))
            mask = np.pad(mask, ((0, 0), (r, l)))
        
        if self.classes > 1:
            one_hot = np.zeros([self.classes+1, self.img_size, self.img_size])
            for i in range(self.classes):   
                one_hot[i] = (mask == i)
            mask = one_hot
        
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.as_tensor(mask)
        if self.transforms is not None:
            image, mask = self.transform(image, mask)
            return image, mask 
        
        return image, mask 
    
    def resize(self, img, resize_to):
        w, h = img.size 
        scale = max(w, h)/resize_to
        if w > h:
            h = int(h//scale)
            img = img.resize((self.img_size, h))
        elif w == h:
            img = img.resize((self.img_size, self.img_size))
        else:
            w = int(w//scale)
           
            img = img.resize((w, self.img_size))
        return img 
    
    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    # check 
    path = 'C:\\Users\\samue\\Downloads\\Senior_Project\\Pytorch-UNet\\data'
    dataset = Dataset(path, classes=3)
    image, mask = dataset[1]
    dataloader = DataLoader(dataset, 16, shuffle=True)
    print('Dataset: ', image.shape, mask.shape)
    image, mask = next(iter(dataloader))
    print('Dataloader, Batch-size = 16: ', image.shape, mask.shape)
