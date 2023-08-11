import config
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from model import YOLOv3
from tqdm import tqdm
import warnings

import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import random
from lightning_utils import get_loader, criterion, accuracy_fn, get_datasets

import seaborn as sn 
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


class YOLOv3Lightning(pl.LightningModule):
  def __init__(self, dataset, lr=config.LEARNING_RATE):
    super().__init__()
    self.model = YOLOv3(num_classes=config.NUM_CLASSES)
    self.lr = lr
    self.scaled_anchors = 0
    self.criterion = criterion
    self.losses = []
    self.threshold = config.CONF_THRESHOLD
    self.iou_threshold = config.NMS_IOU_THRESH
    self.train_idx = 0
    self.box_format="midpoint"
    self.anchors = config.ANCHORS
    self.dataset = dataset
    self.criterion = criterion
    self.accuracy_fn = accuracy_fn
    self.tot_class_preds, self.correct_class = 0, 0
    self.tot_noobj, self.correct_noobj = 0, 0
    self.tot_obj, self.correct_obj = 0, 0

  def forward(self, x):
    return self.model(x)

  def on_train_epoch_start(self):
      # Set a new image size for the dataset at the beginning of each epoch
      size_idx = random.choice(range(len(config.IMAGE_SIZES)))
      self.dataset.set_image_size(size_idx)
      self.scaled_anchors = (
          torch.tensor(config.ANCHORS)
          * torch.tensor(config.S[size_idx]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
      
  def on_validation_epoch_start(self):
      self.scaled_anchors = (
          torch.tensor(config.ANCHORS)
          * torch.tensor(config.S[1]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
      )

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=self.lr,
        steps_per_epoch=1035,
        epochs=self.trainer.max_epochs,
        pct_start=5/self.trainer.max_epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    return [optimizer], [scheduler]


  def training_step(self, batch, batch_idx):
    x, y = batch
    out = self(x)
    loss = self.criterion(out, y, self.scaled_anchors)

    self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
    #self.evaluate(x, y, out, 'train')
    
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, labels = val_batch
    out = self(x)

    loss = self.criterion(out, labels, self.scaled_anchors)
    self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    self.evaluate(x, labels, out, 'val')


  def evaluate(self, x, y, out, stage=None):

    # Class Accuracy 
    class_accuracy, no_obj_accuracy, obj_accuracy = self.accuracy_fn(y, 
                                                                     out, 
                                                                     self.threshold, 
                                                                     self.correct_class, 
                                                                     self.correct_obj, 
                                                                     self.correct_noobj, 
                                                                     self.tot_class_preds, 
                                                                     self.tot_obj,
                                                                     self.tot_noobj, )
    if stage:
      self.log(f'{stage}_class_accuracy', class_accuracy, prog_bar=True, on_epoch=True, on_step=True, logger=True)
      self.log(f'{stage}_no_obj_accuracy', no_obj_accuracy, prog_bar=True, on_epoch=True, on_step=True, logger=True)
      self.log(f'{stage}_obj_accuracy', obj_accuracy, prog_bar=True, on_epoch=True, on_step=True, logger=True)



trainer = pl.Trainer(precision=16,
                     log_every_n_steps=1,
                     check_val_every_n_epoch=1,
                     enable_model_summary=True,
                     max_epochs=config.NUM_EPOCHS,
                     accelerator='auto',
                     devices=1 if torch.cuda.is_available() else None,
                     logger=[CSVLogger(save_dir="logs/"),
                             TensorBoardLogger("logs/", name="YoloV3")],
                     callbacks=[LearningRateFinder(0.0003, 1, 200),
                                LearningRateMonitor(logging_interval="step"),
                                TQDMProgressBar(refresh_rate=10)],
                     )




def main():

  train_dataset, test_dataset = get_datasets()
  train_loader, test_loader = get_loader(train_dataset, test_dataset)

  model = YOLOv3Lightning(train_dataset)
   
  trainer.fit(model, train_loader, test_loader)
  trainer.test(model, test_loader)

  metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
  del metrics["step"]
  metrics.set_index("epoch", inplace=True)
  print(metrics.dropna(axis=1, how="all").head())
  sn.relplot(data=metrics, kind="line")
  plt.show()


if __name__ == "__main__":
    main()