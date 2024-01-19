import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
import nibabel as nib
import cv2


model = TumorSegmentation.load_from_checkpoint("../logs/..")
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device);

subject = Path("../imagesTs/lung_002.nii.gz")
ct = nib.load(subject).get_fdata() / 3071
ct = ct[:,:,30:]

segmentation = []
label = []
slices = []

for i in range (ct.shape[-1]):
    slice = ct[:,:,i]
    slice = cv2.resize(slice, (256,256))
    slice = torch.tensor(slice)
    slices.append(slice)
    
    slice = slice.unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        predict = model(slice)[0][0].cpu()
    predict = predict > 0.5
    segmentation.append(predict)
    label.append(segmentation)

fig = plt.figure()
camera = Camera(fig)

for i in range(0, len(slices), 2):
    plt.imshow(slices[i], cmap='bone')
    mask = np.ma.masked_where(segmentation[i]==0, segmentation[i])
    plt.imshow(mask, alpha=0.5, cmap='autumn')
    plt.axis('off')

    camera.snap()
animation = camera.animate()

HTML(animation.to_html5_video())