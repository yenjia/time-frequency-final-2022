from math import pi
from skimage.util import random_noise
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
    
class Noised(BaseTransform):
    '''
    This transformation supports two types of noise: Gaussian and Impulse noise 
    (Salt and Pepper noise). The parameter "severty" is the same as 
    https://github.com/hendrycks/robustness, which applies several "corruptions"
    to create the ImageNet-C.
    Args:
        mode: the noise mode, defaults to "gaussian"
        serverty: number from 0 to 4, defaults to 0
        guassian: predefined serverty
        impluse: predifined serverty
    '''
    def __init__(self, keys, mode="gaussian", severty=0, **kwargs):
        super(Noised, self).__init__(keys, **kwargs)
        self.mode = mode
        self.severty = severty
        self.gaussian = [0.08, 0.12, 0.18, 0.26, 0.38]
        self.impluse = [0.03, 0.06, 0.09, 0.17, 0.27]
    
    def _process(self, single_data, **kwargs):
        if self.mode == "gaussian":
            img = single_data + torch.randn(single_data.size())*self.gaussian[self.severty]
            return torch.clip(img, 0, 1)
        elif self.mode == "s&p":
            img = random_noise(single_data, mode=self.mode, amount=self.impluse[self.severty])
            return torch.from_numpy(img)


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
        # Normalized(keys=["image"]),
    ]
    if "noise" in transforms_config and isinstance(transforms_config["noise"], dict):
        transforms += [Noised(keys=["image"], **transforms_config["noise"])]

    return Compose(transforms)

def val_transforms(**transforms_config):
    img_size = transforms_config.get("size", 224)
    transforms = [
        LoadImaged(keys=["image"]),
        Resized(keys=["image"], size=(img_size, img_size)),
        Scale01d(keys=["image"]),
        # Normalized(keys=["image"]),
    ]
    if "noise" in transforms_config and isinstance(transforms_config["noise"], dict):
        transforms += [Noised(keys=["image"], **transforms_config["noise"])]

    return Compose(transforms)
