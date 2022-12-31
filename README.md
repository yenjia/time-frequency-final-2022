# Final Project
## Setting the Environment
```
conda create --name final python==3.9.15
conda activate final
pip install -r requirements.txt
```

## How to train the model
You can train the model by using a config.
```
train.py -c ../config/{YOUR CONFIG}
```

### The Structure of Config
```
name: resnet50 (Related to folder name of log and checkpoint) 
savepath: v0 (Related to folder name of log and checkpoint)
epoches: 100
strategy: ddp
accumulate_batches: 8
precision: 16
gradient_clip_val: 5.0
data_config:
  batch_size: 64
  val_batch_size: 64
  dataroot: /neodata/pathology_breast/caltech/256_ObjectCategories/ (Dataroot is expected to be linked to the front of the image path in datalist)
  datalist: /neodata/pathology_breast/caltech/project/prepare/datalist.json
  cache: false
  transforms_config:
    img_size: 224
    random_transform: false
    noise:  (Choose the type of noise)
      mode: gaussian
      severty: 3
model_config:
  model:
    num_classes: 256
    backbone: resnet50
    backbone_num_features: 1000
  loss: FL (Focal loss)
  optimizer: adamw
  lr: 0.0001
  scheduler:
    name: warmcosine
    warmup_epochs: 5
    max_epochs: 100
    eta_min: 0
```

### The Structure of Datalist
Datalist is a JSON file, which is a `dict` format. The `dict` has 3 keys: "training", "validation" and "test". The value corresponding to each key is `list`, which has many `dict`, and each `dict` contains the image path and label.

```
{
    "training": [
        {
            "image": "193.soccer-ball/193_0092.jpg",
            "label": 192
        },
        {
            "image": "253.faces-easy-101/253_0314.jpg",
            "label": 252
        },
        ...
```

## Notes
The model uses the architecture provided by:
* https://github.com/LiQiufu/WaveCNet
* https://github.com/YehLi/ImageNetModel

## Contact
Questions about the actions of this code can be directed to: rex19981002@gmail.com

## Citation
```
@misc{
    title  = {time-frequency-final-2022},
    author = {Yen-Jia, Chen},
    url    = {https://github.com/yenjia/time-frequency-final-2022},
    year   = {2022}
}
```