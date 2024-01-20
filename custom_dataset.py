import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class LungDataset(Dataset):
    def __init__(self, root_path, augments):
        self.slice_files = self.extract_files(root_path)
        self.augments = augments

    @staticmethod
    def extract_files(root_path):
        slice_files = []
        for subject in root_path.glob('*'):
            slice_path = subject/'slices'
            for slice in slice_path.glob('*'):
                slice_files.append(slice)
        return slice_files

    @staticmethod
    def get_mask(path):
        words = list(path.parts)
        words[words.index('slices')] = 'masks'
        return Path(*words)

    def augment(self, slice, mask):
        random_seed = torch.randint(1, 100000, (1,))[0].item()
        imgaug.seed(random_seed)

        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augments(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        return slice_aug, mask_aug
                
    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        file_path = self.slice_files[idx]
        mask_path = self.get_mask(file_path)
        slice = np.load(file_path)
        mask = np.load(mask_path)

        if self.augments:
            slice, mask = self.augment(slice, mask)

        return np.expand_dims(slice,0), np.expand_dims(mask,0)  


""" View an augmented slice and mask
augments = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15),
               rotate=(-45, 45)),
    iaa.ElasticTransformation()
])
path = Path('../preprocessed/train/')
dataset = LungDataset(path, augments)


fig, axis = plt.subplots(3,3,figsize=(9,9))
for i in range(3):
    for j in range(3):
        slice, mask = dataset[120]
        mask_ = np.ma.masked_where(mask==0, mask)
        axis[i][j].imshow(slice[0], cmap='bone')
        axis[i][j].imshow(mask_[0], cmap='autumn')
        axis[i][j].axis('off')

plt.tight_layout()
"""
