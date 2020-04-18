# FARICH-pics

[![Build Status](https://travis-ci.com/82492749123082/farich-pics.svg?branch=dev)](https://travis-ci.com/82492749123082/farich-pics)
[![Quick run in Colab (dev-ветка)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/82492749123082/farich-pics/blob/dev/notebooks/CirclesNN.ipynb) 

Rings detection for FARICH

## Neural network (nn)

We have trained nn to detect rings with [this tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

Dataset contains **100x100px** boards with **one ring** on each and uniform **noise 1%**.

Technical details you can see in [the notebook](notebooks/CirclesNN.ipynb)
[![Quick run in Colab (dev-ветка)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/82492749123082/farich-pics/blob/dev/notebooks/CirclesNN.ipynb) 

### Results

Precision and recall vs treshold predictor score:
![superiority](page/score.png)

Some detection examples

Good/perfect examples:
![good guy](page/TP1.png)
![good guy](page/TP2.png)
![good guy](page/TP3.png)
![good guy](page/TP4.png)

Bad example:
![bad guy](page/FN1.png)

Most problems from such images but even we cannot detect there rings. Why should nn?

## NN for many circles

We also trained nn for many circles (see [notebook](notebooks/ManyCirclesNN.ipynb))
[![Quick run in Colab (dev-ветка)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/82492749123082/farich-pics/blob/dev/notebooks/ManyCirclesNN.ipynb)

### Results
Precision and recall vs treshold predictor score:
![superiority](page/score_many_circles.png)
 

