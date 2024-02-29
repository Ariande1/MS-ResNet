# Advancing Spiking Neural Networks towards Deep Residual Learning

This repo **covers the implementation of the following paper:**

"Advancing Spiking Neural Networks towards Deep Residual Learning". [Paper](https://ieeexplore.ieee.org/abstract/document/10428029).

The most straightforward way of training higher quality models is by increasing their size. In this work, we would like to see that deepening network structures could get rid of the degradation problem and always be a trustworthy way to achieve satisfying accuracy for the direct training of SNNs.

This repository contains the source code for the training of our MS-ResNet on ImageNet. The models are defined in `models/MS_ResNet.py` .

The pretrained models on ImageNet are available at https://drive.google.com/drive/folders/1NMn6RNMpqi6NqKckgFg0t7ewbct-H6_s?usp=sharing.

## Running

1. Install Python 3.7, PyTorch 1.8 and Tensorboard.  

2. Change the data paths `vardir,traindir` to the image folders of ImageNet dataset.

3. To train the model, please run  `CUDA_VISIBLE_DEVICES=GPU_IDs python -m torch.distributed.launch --master_port=1234 --nproc_per_node=NUM_GPU_USED train_amp.py -net resnet34 -b 256 -lr 0.1` .

   `-net` option supports `resnet18/34/104` .

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@ARTICLE{10428029,
  author={Hu, Yifan and Deng, Lei and Wu, Yujie and Yao, Man and Li, Guoqi},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Advancing Spiking Neural Networks Toward Deep Residual Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Degradation;Training;Task analysis;Neurons;Neuromorphics;Computer architecture;Computational modeling;Degradation problem;neuromorphic computing;residual neural network;spiking neural network (SNN)},
  doi={10.1109/TNNLS.2024.3355393}}
```
