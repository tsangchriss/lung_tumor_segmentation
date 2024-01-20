# Automated Lung Tumor Segmentation using Deep Learning

### Preprocessing
- The data contains 96 3D volumes (64 training and 32 testing) CT scans of different lungs with tumors. Each CT scan has a corresponding label (a binary mask) that contains the area of the lung tumor. 
- For memory efficiency, I extracted cross sectional images (slices) of each 3D CT scan, amounting to about 14,500 total slices. Each slice is 512x512 resolution.
- Each pixel in a CT scan slice corresponds to a Houndfield Unit (HU). These values measure how much a part of the body absorbs x-rays. Different tissues absorb x-rays differently, which is why they appear differently on a CT scan. The scale ranges from -1000 HU (less dense body parts) to about 3017 HU (more dense body parts).
The HU values from the data were normalized to a range from -0.33 to 1 to improve the model's computational processing. By normalzing these values, the denser tumor tissues becomes more clear. Higher values indicate denser tissues (like tumors), while lower values imply less dense tissues.
- The slices and masks were resized to 256x256 resolution. For mask resizing, I used the nearest neighbor interpolation to maintain the original masks' shapes and boundaries of the lung tumor.
- 58 CT scans to training and 6 to the validation set.


### Data Augmentations and Custom Dataset
- I applied augmentations to the data to increase the diversity of the dataset so that the model is less likely to overfit (does not learn general patterns) the training set. The augmentations I applied included:
  -  scaling
  -  rotations
  -  elastic transformations (different parts of the image are shifted in different directions by some amount).
- The custom dataset ensures any augmentations applied to a slice are also applied to its mask.
- 
### UNet Model
- 
