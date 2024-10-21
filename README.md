# [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

<p align="left">
<a href="https://arxiv.org/abs/2303.16900" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2203.16900-b31b1b.svg?style=flat" /></a>
<a href="https://colab.research.google.com/drive/1-CAPm6FNKYRbe_lAPxIBxsIH4xowgfg8?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

This is a PyTorch implementation of InceptionNeXt proposed by our paper "[InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)". Many thanks to [Ross Wightman](https://github.com/rwightman), InceptionNeXt is integrated into [timm](https://github.com/huggingface/pytorch-image-models).

![InceptionNeXt](https://user-images.githubusercontent.com/15921929/228630174-1d31ac66-174b-4014-9f6a-b7e6d46af958.jpeg)
**TLDR**: To speed up ConvNeXt, we build InceptionNeXt by decomposing the large kernel depthwise convolution with Inception style. **Our InceptionNeXt-T enjoys both ResNet-50’s speed and ConvNeXt-T’s accuracy.**


## Requirements
Our models are trained and tested in the environment of PyTorch 1.13, NVIDIA CUDA 11.7.1 and timm 0.6.11 (`pip install timm==0.6.11`). If you use docker, check [Dockerfile](docker/Dockerfile) that we used.


Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Models
### InceptionNeXt trained on ImageNet-1K
| Model | Resolution | Params | MACs | Train throughput | Infer. throughput | Top1 Acc |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |  :---:  |
| resnet50 | 224 | 26M | 4.1G | 969 | 3149 | 78.4 |
| convnext_tiny | 224 | 29M | 4.5G | 575 | 2413 | 82.1 |
| [inceptionnext_tiny](https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth) | 224 | 28M | 4.2G | 901 | 2900 | 82.3 |
| [inceptionnext_small](https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth) | 224 | 49M | 8.4G | 521 | 1750 | 83.5 |
| [inceptionnext_base](https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth) | 224 | 87M | 14.9G | 375 | 1244 |  84.0 |
| [inceptionnext_base_384](https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth) | 384 | 87M | 43.6G | 139 | 428 | 85.2 |

### ConvNeXt variants trained on ImageNet-1K
| Model | Resolution | Params | MACs | Train throughput | Infer. throughput | Top1 Acc |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |  :---:  |
| resnet50 | 224 | 26M | 4.1G | 969 | 3149 | 78.4 | - |
| convnext_tiny | 224 | 29M | 4.5G | 575 | 2413 | 82.1 | - |
| [convnext_tiny_k5](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k5.pth) | 224 | 29M | 4.4G | 675 | 2704 | 82.0 |
| [convnext_tiny_k3](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3.pth) | 224 | 28M | 4.4G | 798 | 2802 | 81.5 |
| [convnext_tiny_k3_par1_2](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_2.pth) | 224 | 28M | 4.4G |  818 | 2740 | 81.4 |
| [convnext_tiny_k3_par3_8](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par3_8.pth) | 224 | 28M | 4.4G |  847 | 2762 | 81.4 |
| [convnext_tiny_k3_par1_4](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_4.pth) | 224 | 28M | 4.4G | 871 | 2808 | 81.3 |
| [convnext_tiny_k3_par1_8](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_8.pth) | 224 | 28M | 4.4G | 901 | 2833 | 80.8 |
| [convnext_tiny_k3_par1_16](https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_16.pth) | 224 | 28M | 4.4G | 916 | 2846 | 80.1 |

The throughputs are measured on an A100 with full precision and batch size of 128. See [Benchmarking throughput](#benchmarking-throughput).

#### Usage
We also provide a Colab notebook which run the steps to perform inference with InceptionNeXt: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-CAPm6FNKYRbe_lAPxIBxsIH4xowgfg8?usp=sharing)


## Validation

To evaluate our CAFormer-S18 models, run:

```bash
MODEL=inceptionnext_tiny
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained
```

## Benchmarking throughput
On the environment described above, we benchmark throughputs on an A100 with batch size of 128. The better results of "Channel First" and "Channel Last" memory layouts are reported.

For Channel First:
```bash
MODEL=inceptionnext_tiny # convnext_tiny
python3 benchmark.py /path/to/imagenet  --model $MODEL
```

For Channel Last:
```bash
MODEL=inceptionnext_tiny # convnext_tiny
python3 benchmark.py /path/to/imagenet  --model $MODEL --channel-last
```

## Train
We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.


```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/inceptionnext # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=inceptionnext_tiny # inceptionnext_small, inceptionnext_base
DROP_PATH=0.1 # 0.3, 0.4


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH
```
Training (fine-tuning) scripts of other models are shown in [scripts](/scripts/).


## BibTeX
```
@article{yu2023inceptionnext,
  title={InceptionNeXt: when inception meets convnext},
  author={Yu, Weihao and Zhou, Pan and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2303.16900},
  year={2023}
}
```

## Acknowledgment
Weihao Yu would like to thank TRC program and GCP research credits for the support of partial computational resources. Our implementation is based on [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [metaformer](https://github.com/sail-sg/metaformer).
