## Optimizing Neural Networks with Kronecker-factored Approximate Curvature

* Link: https://arxiv.org/abs/1503.05671

-----

### Introduction

The authors introduce an effective practical approximation of inverse of the Fisher information matrix. That allows to perform 
[Natural Gradient descent](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf) effectively.

K-FAC differs from other popular Hessian-Free optimization methods in a way that it does not involve running Conjugate Gradient
algorithm in order to compute `H^-1 * grad` (which is computationally expensive in practice). K-FAC can converge `>3.5x` faster
in `>14x` fewer iterations than SGD with Momentum 
(numbers are taken from tensorflow [implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/kfac) doc).


### General idea of factorizing the Fisher matrix


%%% TODO
1. Definition of Fisher
2. Rearranging model parameters vector as `Theta = [Theta_1, ..., Theta_l]` (`l` -- number of layers).
3. Each block in resulted `l x l` block matrix is factorized using several Kronecker-product properties (intuition of the
approximation -- activations are not correlated with gradients).
4. Approximating `F^-1` with block-diagonal.
5. Approximating `F^-1` with block-tridiagonal.
6. Estimating `E[a_i a_j^T]` and `E[g_i g_j^T]`.
7. Damping schemes (Levenberg-Marquardt, Tikhonov...)
