# Object Detection with YOLOv3

This repository contains the implementation of YOLOv3 for object detection, enhanced with Grad-CAM visualizations and trained with advanced techniques like multi-resolution training and Mosaic Augmentation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [LR Finder](#finder)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Sample Predictions](#sample-predictions)
- [Contributing](#contributing)
- [License](#license)

## Introduction

YOLOv3 is one of the most popular object detection algorithms. This implementation not only provides accurate object detection capabilities but also visualizes the regions of interest in the image using Grad-CAM.

## Features
**PyTorch Lightning Implementation** 
The codebase is refactored using PyTorch Lightning, providing a cleaner and more maintainable structure. 

```python
class YOLOv3Lightning(pl.LightningModule):
  def __init__(self, dataset=None, lr=config.LEARNING_RATE):
    super().__init__()

    self.save_hyperparameters()

    self.model = YOLOv3(num_classes=config.NUM_CLASSES)
    self.lr = lr
    self.criterion = criterion
    self.losses = []
    self.threshold = config.CONF_THRESHOLD
    self.iou_threshold = config.NMS_IOU_THRESH
    self.train_idx = 0
    self.box_format="midpoint"
    self.dataset = dataset
    self.criterion = criterion
    self.accuracy_fn = accuracy_fn
    self.get_evaluation_bboxes = get_evaluation_bboxes
    self.tot_class_preds, self.correct_class = 0, 0
    self.tot_noobj, self.correct_noobj = 0, 0
    self.tot_obj, self.correct_obj = 0, 0
    self.scaled_anchors = 0

  def forward(self, x):
    return self.model(x)

  def set_scaled_anchor(self, scaled_anchors):
      self.scaled_anchors = scaled_anchors

  def on_train_epoch_start(self):
      # Set a new image size for the dataset at the beginning of each epoch
      size_idx = random.choice(range(len(config.IMAGE_SIZES)))
      self.dataset.set_image_size(size_idx)
      self.set_scaled_anchor((
          torch.tensor(config.ANCHORS)
          * torch.tensor(config.S[size_idx]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
      ))

  def on_validation_epoch_start(self):
      self.set_scaled_anchor((
          torch.tensor(config.ANCHORS)
          * torch.tensor(config.S[1]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
      ))



  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=1e-4,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
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

```

**Multi-resolution Training**
The model is trained on varying resolutions for better generalization.

```python
DIV = 32
IMAGE_SIZES = [416, 608, 896, 1280]
S = [[x//DIV, x//DIV*2, x//DIV*4] for x in IMAGE_SIZES]
```

**Mosaic Augmentation**
A data augmentation technique introduced in the YOLOv4 paper, specifically designed to improve the performance of object detection models. 

**How it works**

- Selection: Four random images are selected from the training dataset.

- Cropping and Resizing: Each of these images is randomly cropped. The cropped regions are then resized to half the width and half the height of the target input size of the model.

- Combining: These resized images are then combined into a single image in a 2x2 grid, forming a mosaic. This results in a new image that contains objects from all four original images.

It allows the model to see multiple scales, aspect ratios, and combinations of objects in a single image. This can be especially beneficial for detecting smaller objects and understanding the context between different objects.

![image](https://github.com/Delve-ERAV1/S13/assets/11761529/081b3c28-e207-438d-9774-33889404b5f0)


**Grad-CAM Visualization**
Visualize where the model is looking in the image to make its predictions.

**HuggingFace Spaces Integration**

```bash
git clone https://github.com/Delve-ERAV1/S13.git
cd S13
pip install -r requirements.txt
gradio app.py
```
The trained model App is hosted on HuggingFace Spaces, allowing users to upload custom images for predictions. App may be accessed [here](https://huggingface.co/spaces/Sijuade/YOLOV3-GradCAM). 

## LR Finder
![image](https://github.com/Delve-ERAV1/S13/assets/11761529/cc9c2b11-52fc-4e7e-9ff5-54b5c08170ea)

# Training

```bash
git clone https://github.com/Delve-ERAV1/S13.git
cd S13
pip install -r requirements.txt
python main.py
```

Training logs may be found [here](training_logs.txt)

![pred1](https://github.com/Delve-ERAV1/S13/assets/11761529/df995d26-8d1b-44cd-8979-df4fd514ed44)
![pred2](https://github.com/Delve-ERAV1/S13/assets/11761529/c343787c-1d39-44f6-86f5-c8c228e193e8)

# Validation
Validation logs may be found [here](validate_logs.txt)


## References
https://arxiv.org/abs/1804.02767 \
https://www.youtube.com/watch?v=Grir6TZbc1M \
https://github.com/jacobgil/pytorch-grad-cam
