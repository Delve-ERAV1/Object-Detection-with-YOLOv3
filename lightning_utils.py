
from torch.utils.data import Dataset, DataLoader
from loss import YoloLoss
import config
import torch
from dataset import YOLODataset

def criterion(out, y, anchors):
  loss_fn = YoloLoss()
  loss = (
          loss_fn(out[0], y[0], anchors[0])
          + loss_fn(out[1], y[1], anchors[1])
          + loss_fn(out[2], y[2], anchors[2]))
  return loss


def get_loader(train_dataset, test_dataset):
  train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=config.BATCH_SIZE,
          num_workers=config.NUM_WORKERS,
          pin_memory=config.PIN_MEMORY,
          shuffle=True,
          drop_last=False,
  )

  test_loader = DataLoader(
      dataset=test_dataset,
      batch_size=config.BATCH_SIZE,
      num_workers=config.NUM_WORKERS,
      pin_memory=config.PIN_MEMORY,
      shuffle=False,
      drop_last=False,
  )

  return(train_loader, test_loader)


def accuracy_fn(y, out, threshold, 
                correct_class, correct_obj, 
                correct_noobj, tot_class_preds, 
                tot_obj, tot_noobj):

  for i in range(3):
      
      obj = y[i][..., 0] == 1 # in paper this is Iobj_i
      noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

      correct_class += torch.sum(
          torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
      )
      tot_class_preds += torch.sum(obj)

      obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
      correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
      tot_obj += torch.sum(obj)
      correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
      tot_noobj += torch.sum(noobj)

  return((correct_class/(tot_class_preds+1e-16))*100, 
         (correct_noobj/(tot_noobj+1e-16))*100, 
         (correct_obj/(tot_obj+1e-16))*100)


def get_datasets(train_loc="/train.csv", test_loc="/test.csv"):

  train_dataset = YOLODataset(
      config.DATASET + train_loc,
      transform=config.train_transform,
      img_dir=config.IMG_DIR,
      label_dir=config.LABEL_DIR,
      anchors=config.ANCHORS,
  )

  test_dataset = YOLODataset(
      config.DATASET + test_loc,
      transform=config.test_transform,
      img_dir=config.IMG_DIR,
      label_dir=config.LABEL_DIR,
      anchors=config.ANCHORS,
  )

  return(train_dataset, test_dataset)