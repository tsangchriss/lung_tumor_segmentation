# Automated Lung Tumor Segmentation using Deep Learning

### Preprocessing
- The data contains 96 3D volumes (64 training and 32 testing) CT scans of different lungs with tumors. Each CT scan has a corresponding label (a mask) that contains the area of the lung tumor. 
- For memory efficiency, I extracted cross sectional images (slices) of each 3D CT scan, amounting to about 14,500 total slices. Each slice is 512x512 resolution.
- Each pixel in a CT scan slice corresponds to a Houndfield Unit (HU). These values measure how much a part of the body absorbs x-rays. Different tissues absorb x-rays differently, which is why they appear differently on a CT scan. The scale ranges from -1000 HU (less dense body parts) to about 3017 HU (more dense body parts).
The HU values from the data were normalized to a range from -0.33 to 1 to improve the model's computational processing. By normalzing these values, the denser tumor tissues becomes more clear. Higher values indicate denser tissues (like tumors), while lower values imply less dense tissues.
- The slices and masks were resized to 256x256 resolution. For mask resizing, I used the nearest neighbor interpolation to maintain each of the mask's original shapes and boundaries of the lung tumor.
- 58 CT scans for training and 6 for validation.

<div align="center">
  
https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/f4171c4f-2f9f-4392-bd4e-cb1f8ec028bf

</div>





### Data Augmentations and Custom Dataset
- A custom dataset ensures any augmentations applied to a slice are also applied to its mask.
- I applied augmentations to the masks to increase the diversity of the dataset so that the model is less likely to overfit (does not learn general patterns) the training set. The augmentations I applied include:
  -  scaling
  -  rotations
  -  elastic transformations (different parts of the image are shifted in different directions by some amount).

<div align="center">
    <p>Mask Augmentation Example</p>
    <img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/37ad452a-fd74-4eca-b2c2-3895f674bc56" width="500" height="500">
</div>

### UNet Model
- A UNet model is well suited for this segmentation task because it captures the borders of the lung tumor during the encoder phase, and ensures that small details are not lost while reconstructing the image during the decoder phase (skip connections).

### Imbalanced Data and Training Loop
- Of the roughly 14,500 slices, only about 1500 contain a tumor revealing an inbalanced training set. Left unchecked, the model might learn to predict a CT scan has no tumor because the majority of its slices does reveal a tumor. To combat this, I used oversampling to assign a larger weight to slices with tumors. This weight is the ratio of slices without tumors to slices with tumors (~8.5).
- The training loop uses a binary cross entropy loss function and Adam optimizer function (lr=1e4).
- Training duration was about 7 hours on an RTX 2070.
  
### Results
Training Loss (included smoothing to make it easier to see):
<img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/4762723a-c770-4518-91a2-2e0abb6407df" width="870" height="500">


Validation Loss (included smoothing to make it easier to see):
<img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/447ff452-fcdc-4c96-a802-11ab298d0507" width="870" height="500">

A predicted mask example (passed a CT scan the model has never seen before).

<div align="center">
  
https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/3417fecd-aba6-45ab-844e-0963884c3eac

</div>








