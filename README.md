### Contents
- [Installation](#Installation)
- [Datasets](#Datasets)
- [Training model](#Training model)

### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2)

Install the required packages:
```
python==3.8.1
numpy==1.22.3
torch==1.10.1
torchvision==0.11.2
detectron2==0.6
kornia==0.6.3
clip==1.0      git+https://github.com/openai/CLIP.git
pymage_size==1.4.1
opencv-python
setuptools==58.0.4
Pillow==9.5.0
```
### Datasets
Set the environment variable DETECTRON2_DATASETS in [file](./data/datasets/builtin.py) to the parent folder of the datasets

```
path-to-parent-dir/
    /Diverse-Weather
        /daytime_clear
        /daytime_foggy
        /dusk_rainy
        /night_rainy
        /night_sunny
```
Download [Diverse Weather](https://github.com/AmingWu/Single-DGOD) Datasets and place in the structure as shown.


### Training model
Run the following command to train the model:
```
sh train.sh
```
