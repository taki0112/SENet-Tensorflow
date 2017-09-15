# SENet-Tensorflow
Simple Tensorflow implementation of ***![Squeeze Excitation Networks](https://arxiv.org/abs/1709.01507)*** using **Cifar10** 

I implemented the following SENet
* ![ResNeXt](https://arxiv.org/abs/1611.05431)
* ![Inception-v4, Inception-resnet-v2](https://arxiv.org/abs/1602.07261)

If you want to see the ***original author's code***, please refer to this [link](https://github.com/hujie-frank/SENet)



## Requirements
* Tensorflow 1.x
* Python 3.x
* tflearn (If you are easy to use ***global average pooling***, you should install ***tflearn***)

## What is the "SE block" ?
![senet](./assests/senet_block.JPG)

## How apply ? (Inception, Residual)
<div align="center">
   <img src="https://github.com/hujie-frank/SENet/blob/master/figures/SE-Inception-module.jpg" width="420">
  <img src="https://github.com/hujie-frank/SENet/blob/master/figures/SE-ResNet-module.jpg"  width="420">
</div>

## How "Reduction ratio" should I set?
![reduction](./assests/reduction_ratio.JPG)
* **original** refers to ***ResNet-50***

## Benefits against Network Depth
![depth](./assests/benefit_depth.JPG)

## Incorporation with Modern Architecture
![incorporation](./assests/result2.JPG)

## Comparison with State-of-the-art
![compare](./assests/result.JPG)

## Related works
* [Densenet](https://github.com/taki0112/Densenet-Tensorflow)
* [ResNeXt](https://github.com/taki0112/ResNeXt-Tensorflow)

## Reference
* [Inception_korean](https://norman3.github.io/papers/docs/google_inception.html)

## Author
Junho Kim
