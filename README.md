# Keyframes-GAN (IEEE TMM 2023)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.04263)
[![GitHub Stars](https://img.shields.io/github/stars/LorenzoAgnolucci/Keyframes-GAN?style=social)](https://github.com/LorenzoAgnolucci/Keyframes-GAN)

### [Download the video demo](media/video_demo.mp4)

## Table of Contents

* [Overview](#overview)
* [Prerequisites and Installation](#prerequisites-and-installation)
* [Usage](#usage)
  * [Testing](#testing)
  * [Training](#training)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)

## Overview
![Inference example](media/inference_example.png)

This is the official repo of the paper [***Perceptual Quality Improvement in Videoconferencing using Keyframes-based GAN***](https://ieeexplore.ieee.org/abstract/document/10093128).

In this work we propose a novel GAN architecture for compression artifacts reduction in videoconferencing. In this context,
the speaker is typically in front of the camera and remains the same for the entire duration of the transmission.
With this assumption, we can maintain a set of reference keyframes of the person from the higher quality I-frames that
are transmitted within the video streams. First, we extract multi-scale features from the compressed and reference frames.
Then, these features are combined in a progressive manner with Adaptive Spatial Feature Fusion blocks based on facial
landmarks and with Spatial Feature Transform blocks. This allows to restore the high frequency details lost after the
video compression.

![Architecture](media/architecture.png)

## Prerequisites and Installation
1. Clone the repo
```sh
git clone https://github.com/LorenzoAgnolucci/Keyframes-GAN.git
```


2. Create a virtual env and install all the dependencies with
```sh
pip install -r requirements.txt
```


3. Even if it is not required, we strongly recommend to install ```dlib``` with GPU support


4. For metrics computation, you need to run
```sh
pip install -e pybrisque/
```


5. Download the pretrained models
  * [Model weights](https://drive.google.com/file/d/1JDBgiwEFpBHMtIJLRd1y_9IRKmw99MgN/view?usp=sharing)
  * [Dlib CNN face detector](https://drive.google.com/file/d/1l2R9qImsBXkCgk698v1QUp0k7rqudeRd/view?usp=sharing)
  * [Dlib shape predictor](https://drive.google.com/file/d/1bLLe01Bw8SNdVZIJjBTKriqFNHZlaPQL/view?usp=sharing)
 
and move them inside the ```pretrained_models``` folder

   
## Usage
For testing, you need one or more HQ ```mp4``` videos. These videos will be compressed with a given CRF. The face from each frame
will be cropped, aligned and then restored with our model exploiting HQ keyframes.

### Testing
1. Move the HQ videos under a directory named ```{BASE_PATH}/original/```


2. Run
```sh
python preprocessing.py --base_path {BASE_PATH} --crf 42
```

where ```crf``` is a given Constant Rate Factor (default 42)

3. Run
```sh
python video_inference.py --base_path {BASE_PATH} --crf 42 --max_keyframes 5
```
where ```crf``` must be equal to the one of step 2 and ```max_keyframes``` is the max cardinality of the set of keyframes (default 5)

4. If needed, run
```sh
python compute_metrics.py --gt_path {BASE_PATH}/original --inference_path inference/DMSASFFNet/max_keyframes_5/LFU
```
where ```gt_path``` is the directory that contains the HQ videos and ```inference_path``` is the directory that contains the restored frames

### Training
1. Modify the file ```BasicSR/options/train/DMSASFFNet/train_DMSASFFNet.yml``` to indicate the path of your training and validation datasets

2. Start training by running the following command with `BasicSR` as the current working directory:

```sh
python basicsr/train.py -opt options/train/DMSASFFNet/train_DMSASFFNet.yml
``` 
Please refer to [BasicSR](https://github.com/xinntao/BasicSR) for more information on the fields of the _options_ file.


## Citation

```bibtex

@article{agnolucci2023perceptual,
  title={Perceptual quality improvement in videoconferencing using keyframes-based {GAN}},
  author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  journal={IEEE Transactions on Multimedia},
  volume={26},
  pages={339--352},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgments
We rely on [BasicSR](https://github.com/xinntao/BasicSR) for the implementation of our model and for metrics computation.
