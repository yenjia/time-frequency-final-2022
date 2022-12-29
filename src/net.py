import json
import os
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from models.wavevit import wavevit_s
from models.wresnet import wresnet50
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121, resnet50, swin_t, vit_b_16
from transforms import train_transforms, val_transforms


def get_model(name, weights=None):
    supported_model = {
        "densenet": densenet121(weights=weights),
        "swin_t":swin_t(weights=weights),
        "wavevit_s": wavevit_s(),
        "resnet50": resnet50(weights=weights),
        "wresnet50": wresnet50(),
        "vit": vit_b_16(weights=weights),
    }
    return supported_model[name.lower()]

def get_loss(name):
    supported_loss = {
        "CE": CrossEntropyLoss(),
        "FL" : torch.hub.load('adeelh/pytorch-multi-class-focal-loss', 
                                model='FocalLoss', 
                                gamma=2, 
                                reduction='mean'),
    }
    return supported_loss[name.upper()]

def get_optimizer(name):
    supported_optimizer = {
        "adam": Adam,
        "adamw": AdamW,
    }
    return supported_optimizer[name.lower()]

def get_scheduler(optimizer, scheduler_config):
    supported_scheduler = {
        "cosine": CosineAnnealingLR(optimizer, 
                                    T_max = scheduler_config.get("T_max", 50),
                                    eta_min = scheduler_config.get("eta_min", 0)),
        "warmcosine": LinearWarmupCosineAnnealingLR(optimizer, 
                                                    max_epochs=scheduler_config.get("max_epochs"),
                                                    warmup_epochs=scheduler_config.get("warmup_epochs"))
    }
    return supported_scheduler[scheduler_config["name"].lower()]

class ImageDataset(Dataset):
    def __init__(self, datalist, transform):
        super(ImageDataset, self).__init__()
        self.datalist = datalist
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index].copy()
        return self.transform(data) if self.transform else data

def get_dataset(datalist, transform):
        return ImageDataset(datalist=datalist, transform = transform)

# ----------------------------------------------------------------
class LitModel(LightningModule):
    def __init__(self, config):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.model = get_model(config["model"]["backbone"].lower(), config["model"].get("weights", None))
        
        self.linear = nn.Linear(config["model"]["backbone_num_features"], config["model"]["num_classes"])
        self.loss_function = get_loss(config["loss"])

    def forward(self, x: torch.Tensor) :
        feats = self.model(x)
        y = self.linear(feats)
        return feats, y

    def configure_optimizers(self):
        optimizer =  get_optimizer(self.hparams.config["optimizer"])(
            self.parameters(),
            lr = self.hparams.config["lr"],
        )

        scheduler = get_scheduler(optimizer, self.hparams.config["scheduler"])

        return [optimizer], [scheduler]
    
    def share_step(self, batch : Any, batch_idx : int) :
        images, labels = batch["image"], batch["label"]
        feats, output = self.forward(images)
        labels = labels.long()
        loss = self.loss_function(output, labels)

        preds = torch.argmax(output, dim=1)
        
        result = {"loss" : loss, 
                  "output" : preds.detach(), 
                  "labels" : labels.detach()}
        return result


    def epoch_end(self, outputs, prefix):
        loss_result  = torch.mean(torch.stack([o["loss"] for o in outputs])).detach().cpu()
        y       = torch.cat([o["labels"] for o in outputs]).cpu()
        y_pred  = torch.cat([o["output"].detach() for o in outputs]).cpu()

        acc = accuracy_score(y, y_pred)

        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/loss", loss_result, prog_bar=True)
        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/acc", acc, prog_bar=True)

    def training_step(self, batch : Any, batch_idx : int) -> Any :
        return self.share_step(batch, batch_idx)

    def training_epoch_end(self, outputs : Sequence) -> Any :
        self.epoch_end(outputs,"training")

    def validation_step(self, batch : Any, batch_idx : int) -> Any:
        return self.share_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: Sequence) -> Any :
        self.epoch_end(outputs, "val")

    def test_step(self, batch : Any, batch_idx : int) -> Any:
        return self.share_step(batch, batch_idx, prefix="test")

    def test_epoch_end(self, outputs: Sequence) -> Any :
        self.epoch_end(outputs, "test")
        
    def predict_step(self, batch, batch_idx):
        feats, output = self(batch["image"])
        output = torch.argmax(output, dim=1)
        return {"output": output}

# ----------------------------------------------------------------
def concat_path(datalist: list, dataroot:str):
    subsets = [s for s in ["training", "validation", "test"] if s in datalist]

    for subset in subsets:
        for i in datalist[subset]:
            i["image"] = os.path.join(dataroot, i["image"])

    return datalist

class DataModule(LightningDataModule):
    def __init__(self, **data_config):
        super(DataModule, self).__init__()
        self.batch_size = data_config["batch_size"]
        self.val_batch_size = data_config.get("val_batch_size", 1)
        self.dataroot = data_config["dataroot"]
        self.datalist = data_config["datalist"]
        self.transforms_config = data_config["transforms_config"]

    def setup(self, stage=None): 
        data_list = json.load(open(self.datalist))
        data_list = concat_path(data_list, self.dataroot)

        train_files  = data_list["training"]
        val_files = data_list["validation"]

        self.train_ds = get_dataset(datalist=train_files, transform=train_transforms(**self.transforms_config))
        self.val_ds = get_dataset(datalist=val_files, transform=val_transforms(**self.transforms_config))

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size = self.batch_size,
                          num_workers = 16,
                          shuffle=True,
                          )

    def val_dataloader(self): 
        return DataLoader(self.val_ds,
                          batch_size = self.val_batch_size,
                          num_workers = 16,
                          )
    
