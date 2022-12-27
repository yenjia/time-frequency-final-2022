from math import pi
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import (Resize, 
                                    Normalize, 
                                    AutoAugment, 
                                    RandomRotation, 
                                    Compose)

class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parseVariables(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data
    
    def _parseVariables(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        pass

    def _update_prob(self, cur_ep, total_ep):
        pass

class LoadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadImaged, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        return read_image(path=single_data, mode=ImageReadMode.RGB)

class Scale01d(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Scale01d, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        return single_data.float()/255

class Normalized(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Normalized, self).__init__(keys, **kwargs)
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    def _process(self, single_data, **kwargs):
            return self.normalize(single_data)
        

class Resized(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Resized, self).__init__(keys, **kwargs)
        self.resize = Resize(kwargs.get("size", (224, 224)))

    def _process(self, single_data, **kwargs):
        return self.resize(single_data)

class AutoAugmentd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(AutoAugmentd, self).__init__(keys, **kwargs)
        self.autoaug = AutoAugment()

    def _process(self, single_data, **kwargs):
        return self.autoaug(single_data)
    
def train_transforms(**transforms_config):
    img_size = transforms_config.get("size", 224)
    
    transforms = [
        LoadImaged(keys=["image"]),
        Resized(keys=["image"], size=(img_size, img_size))
    ]
    if transforms_config.get("random_transform", False) == True:
        transforms += [AutoAugmentd(keys=["image"])]
    transforms += [
        Scale01d(keys=["image"]),
        Normalized(keys=["image"]),
    ]

    return Compose(transforms)

def val_transforms(**transforms_config):
    img_size = transforms_config.get("size", 224)
    transforms = [
        LoadImaged(keys=["image"]),
        Resized(keys=["image"], size=(img_size, img_size)),
        Scale01d(keys=["image"]),
        Normalized(keys=["image"]),
    ]

    return Compose(transforms)
