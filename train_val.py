from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


train_path = Path('../train/')
val_path = Path('../val/')

train_dataset = LungDataset(train_path, augments=augments)
val_dataset = LungDataset(val_path, augments=None)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


class TumorSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = UNet()
        self.loss_fn = nn.BCEWithLogitsLoss() 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def forward(self, x):
        predict = self.model(x)
        return predict

    def training_step(self, batch, batch_idx):
        slice, mask = batch
        mask = mask.float()
        predict = self.model(slice.float())
        loss = self.loss_fn(predict, mask)

        self.log('Train Loss', loss)
        if batch_idx % 50 == 0:
            self.log_images(slice.cpu(), mask.cpu(), predict.cpu(), 'Train')
        return loss

    def validation_step(self, batch, batch_idx):
        slice, mask = batch
        mask = mask.float()
        predict = self.model(slice.float())
        loss = self.loss_fn(predict, mask)

        self.log('Valid Loss', loss)
        if batch_idx % 50 == 0:
            self.log_images(slice.cpu(), mask.cpu(), predict.cpu(), 'Val')
        return loss

    def log_images(self, slice, mask, predict, name):

        results = []
        predict = predict > 0.5 # only want output activations greater than 0.5 
        # (sigmoid function will be used to narrow an output activation between 0 and 1)

        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(slice[0][0], cmap='bone')
        mask_ = np.ma.masked_where(mask[0][0]==0, mask[0][0])
        axis[0].imshow(mask_, cmap='autumn')
        axis[0].set_title('Ground Truth')

        axis[1].imshow(slice[0][0], cmap='bone')
        mask_ = np.ma.masked_where(predict[0][0]==0, predict[0][0])
        axis[1].imshow(mask_, cmap='autumn')
        axis[1].set_title('Prediction')

        self.logger.experiment.add_figure(f'{name} Prediction vs Label', fig, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer]


model = TumorSegmentation()
checkpoint = ModelCheckpoint(monitor='Valid Loss', save_top_k=30, mode='min') 
trainer = pl.Trainer(logger=TensorBoardLogger(save_dir='../logs'), log_every_n_steps=1, 
                     callbacks=checkpoint, max_epochs=30, accelerator='gpu', devices=1)

trainer.fit(model, train_loader, val_loader)


%load_ext tensorboard
%tensorboard --logdir ../logs --host localhost