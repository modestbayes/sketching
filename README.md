# Randomized projection for constrained regression

Tensorflow implementation of https://arxiv.org/pdf/1411.0347v1.pdf

## Regression models
* Linear regression
* Logistic regression (unstable numerically)

## Projection methods
* Subsampling
* Gaussian noise
* Hadamard projection

## Mean squared error ratio
| | Subsampling | Gaussian | Hadamard |
| --- | --- | --- | --- |
| r=128 | 14.56 | 13.07 | 12.88 |
| r=256 | 6.26 | 5.92 | 4.82 |
| r=512 | 3.03 | 3.16 | 2.07 |
