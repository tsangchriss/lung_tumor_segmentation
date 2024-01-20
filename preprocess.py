from pathlib import Path
import nibabel as nib
import numpy as np
import cv2

from celluloid import Camera
from IPython.display import HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


image_path = Path('../imagesTr')
label_path = Path('../labelsTr')

image_path_list = list(image_path.glob('lung*'))
example_image_path = image_path_list[0]

def get_label_path(path):
    words = list(path.parts)
    words[words.index('imagesTr')] = 'labelsTr'
    return Path(*words)

""" View an example CT scan and label

example_label_path = get_label_path(example_image_path)
example_image = nib.load(example_image_path)
example_label = nib.load(example_label_path)
nib.aff2axcodes(example_image.affine)

example_image = example_image.get_fdata()
example_label = example_label.get_fdata()

fig = plt.figure()
camera = Camera(fig)

for i in range(example_image.shape[-1]):
    plt.imshow(example_image[:,:,i], cmap='bone')
    mask_ = np.ma.masked_where(example_label[:,:,i]==0, example_label[:,:,i])
    plt.imshow(mask_, alpha=0.5, cmap='autumn')
    plt.axis('off')
    camera.snap()

animation = camera.animate()
HTML(animation.to_html5_video()) 

"""


save_path = Path('../preprocessed')
for subject, image_path in enumerate(tqdm(image_path_list)):
    
    ct_scan = nib.load(image_path).get_fdata()
    label_path = get_label_path(image_path)
    label = nib.load(label_path).get_fdata()

    ct_scan = ct_scan[:,:,30:] / 3071 # crop the first 30 slices because scans starts at the feet.
    label = label[:,:,30:]

    if subject < 57:
        current_save = save_path/'train'/str(subject)
    else:
        current_save = save_path/'val'/str(subject)

    for i in range(ct_scan.shape[-1]):
        slice = ct_scan[:,:,i]
        mask = label[:,:,i]
        slice = cv2.resize(slice,(256,256))
        mask = cv2.resize(mask,(256,256), interpolation=cv2.INTER_NEAREST)

        slice_path = current_save/'slices'
        mask_path = current_save/'masks'
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        np.save(slice_path/str(i), slice)
        np.save(mask_path/str(i), mask) 
