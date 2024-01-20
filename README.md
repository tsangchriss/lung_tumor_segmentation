# Reflection 
This project became the most challening one I've had so far. Before this project, I had a pretty rudementary understanding of Python and engaging with this project made that clear. A lot of my time was spend dedicated to understanding Python and exploring its libraries. I honestly think more time was spent on understanding Python than researching the medical aspects for this project. Something this project has shown me is that planning before writing code will make coding much easier. There were many times that I would rush into a problem by writing for loops and if else statements without realizing there were libraries designed to handle those problems (god bless the creators of nibabel and glob). Ultimately, I think this endeavor improved my coding abilities, drilling into me the essential principles of programming. 

Looking ahead, I want to work with 3D imaging and different applications of convolution neural networks. Although I don't have the GPU or memory capacity, I think that handling more intricate datasets could really help deepen my understanding of segmentation tasks. Also, especially after this project, I'm considering using specialized tools for segmentation like TorchIO.

# Automated Lung Tumor Segmentation using Deep Learning
Lung cancer, or Bronchial Carcinoma, is one of the most common cancers in the United States. More people die from lung cancer than any other type of cancer. According to the CDC, for every 100,000 people 31.8 die from lung and bronchus cancer. Compartively, 18.5% per 100,000 people die from prostate cancer. However, people with lung cancer are living longer after their diagnosis because more cases are found early, when treatment works best.

Automatic tumor segmentation is a method of using deep learning to identify tumors in images. This method can reduce the probability of missing a tumor and identifies the size/volume of the tumor, which is necessary for tumor staging. 

The data can be found here http://medicaldecathlon.com
### Preprocessing
- The data contains 96 3D volumes (64 training and 32 testing) CT scans of different lungs with tumors. Each CT scan has a corresponding label (a mask) that contains the area of the lung tumor. 
- For memory efficiency, I extracted cross sectional images of each 3D CT scan (slices) and mask, amounting to about 30,000 slices and masks.
- The HU values from the data were normalized to a range from -0.33 to 1. By normalizing these values, the tumor tissues become more clear. Higher values indicate denser tissues (like tumors), while lower values imply less dense tissues.
  - Each pixel in a slice corresponds to a Houndfield Unit (HU). These values measure how much a part of the body absorbs x-rays. Different tissues absorb x-rays differently, which is why they appear differently on a CT scan. The scale ranges from -1000 HU (less dense body parts) to about 3017 HU (more dense body parts).
- The slices and masks were resized to 256x256 resolution (originally 512x512). For mask resizing, I used the nearest neighbor interpolation to maintain the shapes and boundaries of the lung tumor.
- 58 CT scans for training and 6 for validation.

<div align="center">
  
https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/f4171c4f-2f9f-4392-bd4e-cb1f8ec028bf

</div>

### Data Augmentations and Custom Dataset
- The custom dataset ensures that whenever a slice undergoes augmentations, the same changes are applied to its corresponding mask.
- These augmenations expand the original dataset to reduce the chances of the model overfitting, such as:
  - Scaling - adjusting the size of the image
  - Rotations - turning the angles of the image
  - Elastic Transformations - shifting different parts of the image in random directions

<div align="center">
    <p>Augmentation Examples</p>
    <img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/37ad452a-fd74-4eca-b2c2-3895f674bc56" width="500" height="500">
</div>

### UNet Model
I followed the UNet Model structure when creating a neural network because of its ability to recognize the boundaries of the lung tumor. Also, it uses skip connections to retain the fine details that might have been lost in the encoder as the image is reconstructed.

### Imbalanced Data and Training Loop
- Out of ~14,500 slices in the dataset, only about 1,500 contain a tumor, indicating an imbalance in the training set. To address this, I used oversampling to assign a larger weight to slices that contain tumors. This weight is the ratio of slices without tumors to those with tumors (~8.5).
  - An imbalance dataset could lead the model to consistently predict the absence of tumors in CT scans, because the majority of slices do not show tumors.
- In the training loop, I used a binary cross entropy loss function along with the Adam optimizer (lr=1e-4)
- It took about 7 hours to complete the loop (RTX 2070 GPU).
  
### Results
Training Loss (included smoothing to make it easier to see):
<img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/4762723a-c770-4518-91a2-2e0abb6407df" width="870" height="500">


Validation Loss (included smoothing to make it easier to see):
<img src="https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/447ff452-fcdc-4c96-a802-11ab298d0507" width="870" height="500">

A predicted mask example (passed a CT scan the model has never seen before).

<div align="center">
  
https://github.com/tsangchriss/lung_tumor_segmentation/assets/137365886/3417fecd-aba6-45ab-844e-0963884c3eac

</div>






