# ScaleKD: Strong Vision Transformers Could Be Excellent Teachers
By Jiawei Fan, Chao Li, Xiaolong Liu and Anbang Yao.

This repository is the official PyTorch implementation 
ScaleKD ([ScaleKD: Strong Vision Transformers Could Be Excellent Teachers](https://openreview.net/pdf?id=0WCFI2Qx85)) published in NeurIPS 2024.

## Introduction

In this paper, we question if well pre-trained vision transformer (ViT) models could be used as teachers that exhibit scalable properties 
to advance cross architecture knowledge distillation research, in the context of adopting mainstream large-scale visual recognition datasets for evaluation. To make this possible, our analysis underlines the importance of seeking effective strategies to align (1) feature computing paradigm differences, (2) model scale differences, and (3) knowledge density differences. By combining three closely coupled components namely *cross attention projector*, *dual-view feature mimicking* and *teacher parameter perception* tailored to address the alignment problems stated above, we present a simple and effective knowledge distillation method, called *ScaleKD*. Our method can train student backbones that span across a variety of convolutional neural network (CNN), multi-layer perceptron (MLP), and ViT architectures on image classification datasets, achieving state-of-the-art knowledge distillation performance. 

![architecture](imgs/teaser.png)
Overview of three core components in our ScaleKD, which are (a) cross attention projector,
(b) dual-view feature mimicking, and (c) teacher parameter perception. Note that the teacher model
is frozen in the distillation process and there is no modification to the student’s model at inference.



## Requirement and Dataset

### Environment
- Python 3.8 (Anaconda is recommended)
- CUDA 11.1
- PyTorch 1.10.1
- Torchvision 0.11.2

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

*Note that using pytorch with higher CUDA version may result in low training speed.*


### Prepare datasets
- Following [this repository](https://github.com/pytorch/examples/tree/main/imagenet#requirements),
- Download the ImageNet dataset from http://www.image-net.org/.
- Then, move and extract the training and validation images to labeled subfolders, using [the following script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).
- Move the data into folder `data/imagenet`



## How to apply ScaleKD to various teacher-student network pairs
Basically, we peform our experiments on two different training strategies. 

### Training with traditional training strategy
- The experiments based on the traditional training strategy are performed on 8 GPUs from a single node.
- Training configurations for various teacher-student network pairs are in folder `configs/distillers/traditional_traning_strategy/`
- Run distillation by following command:
  ```
    bash tools/dist_train.sh $CONFIG_PATH $NUM_GPU
  ```

- Here, we give an example on running `swin-s_distill_res50_img_s3_s4.py` on 8 GPUs:
  ```
  bash tools/dist_train.sh configs/distillers/traditional_traning_strategy/swin-s_distill_res50_img_s3_s4.py 8
  ```

### Training with advanced training strategy
- The experiments based on the advanced training strategy are performed on 32 GPUs from 4 nodes.
- Training configurations for various teacher-student network pairs are in folder `configs/distillers/advanced_traning_strategy/`
- Run distillation by following command:
  ```
    bash run.sh $CONFIG_PATH $NUM_GPU $NODE_RANK
  ```
- Here, we give an example on running `swin-l_distill_res50_img_s3_s4.py` on 32 GPU from 4 nodes (8 GPUs per node):
  ```
  # Node 1
  bash run.sh configs/distillers/advanced_training_strategy/swin-l_distill_res50_img_s3_s4.py 8 0
  # Node 2
  bash run.sh configs/distillers/advanced_training_strategy/swin-l_distill_res50_img_s3_s4.py 8 1
  # Node 3
  bash run.sh configs/distillers/advanced_training_strategy/swin-l_distill_res50_img_s3_s4.py 8 2
  # Node 4
  bash run.sh configs/distillers/advanced_training_strategy/swin-l_distill_res50_img_s3_s4.py 8 2
  ```
- If you want to adapt these experiments to a single node, please adjust the batch size or learning rate accordingly. And then use similar command as above:
  ```
    bash tools/dist_train.sh $CONFIG_PATH $NUM_GPU
  ```




## Transfering checkpoints
```
# Tansfer the Distillation model into mmcls model
python pth_transfer.py --dis_path $dis_ckpt --output_path $new_mmcls_ckpt
```

## Testing

```
#multi GPU
bash tools/dist_test.sh configs/deit/deit-tiny_pt-4xb256_in1k.py $new_mmcls_ckpt 8 --metrics accuracy
```


## Results
<img src="imgs/results.png" width="950px"/>


<!-- |  Model   | Teacher  | T_weight  | Baseline | ViTKD | weight | ViTKD+NKD | weight |                            dis_config                            |
| :------: | :-------: | :-------: | :----------------: | :------------: | :--: | :--: | :--: | :----------------------------------------------------------: |
|   DeiT-Tiny   | DeiT III-Small | [baidu](https://pan.baidu.com/s/1asMuS6E7OmdZzQBH9ugCZg?pwd=83x7)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWHFQNy6OqrZoA82?e=eQ4kmI) |        74.42        |      76.06 (+1.64)      |[baidu](https://pan.baidu.com/s/1OYGeZ2P8RRdEIWM3diyzQA?pwd=niiw)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnVz0irqzX2VP0tg_?e=75Vfs6) |77.78 (+3.36)| [baidu](https://pan.baidu.com/s/1StOAQziPEvvHzQqWvy20vQ?pwd=emct)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnV1cQsVw9SHvSWpG?e=RuE1aL) | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-s3_distill_deit-t_img.py) |
|   DeiT-Small   | DeiT III-Base | [baidu](https://pan.baidu.com/s/15HNMudacNlBUCZ6ySFhENg?pwd=6mmp)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWTTrFh-ST9BcHb8?e=wj3iqH) |        80.55        |      81.95 (+1.40)      |[baidu](https://pan.baidu.com/s/17O64Q4py6Ex1ohjnrPpiew?pwd=4srr)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnV4Fb5EIZEf81PxK?e=K7M1Sz) |83.59 (+3.04)| [baidu](https://pan.baidu.com/s/1OThOyOR60CCxszxB6rY4QQ?pwd=4x90)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnV_tNpvVZ21Yc9eM?e=vlYr8K) | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-b3_distill_deit-s_img.py) |
|   DeiT-Base   | DeiT III-Large | [baidu](https://pan.baidu.com/s/1qdgcTMz_FeBfEH2rchh_yg?pwd=n5hf)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWaR3tslskypZbwB?e=D1aL6p) |        81.76        |      83.46 (+1.70)      |[baidu](https://pan.baidu.com/s/1Qytl5BHpc3qdlYSQq750FQ?pwd=ej2k)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWMMyJZT2NlsIgBg?e=JM5L9h) |85.41 (+3.65)| [baidu](https://pan.baidu.com/s/19Zxq4g3Z1mGhDPjkbG_t0g?pwd=q915)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWJvNpY3Feo_OvGi?e=iPuWJu) | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-l3_distill_deit-b_img.py) | -->



## Citation
```
@article{fan2024scalekd,
  title={ScaleKD: Strong Vision Transformers Could Be Excellent Teachers},
  author={Fan, Jiawei and Li, Chao and Liu, Xiaolong and Yao, Anabang},
  journal={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}
```


## Acknowledgement
This repository is built based on [mmpretrain repository](https://github.com/open-mmlab/mmpretrain) and [cls_KD repository](https://github.com/yzd-v/cls_KD). We thank the authors of the two repositories for releasing their amazing codes.
