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

We want to perform natural gradient updates of form `dTheta = F^-1 * dL/dTheta`, where `Theta` are parameters of neural network, `F` is the Fisher Information matrix corresponding to the distribution `p(y|x, Theta)` which our network models. `L` is the loss function. To compute `F^-1` efficiently, we try to approximate it with the following steps:

1. At first, lets rearrange parameters in `Theta` vector to `l` blocks corresponding to each layer of our feedforward net. So `Theta = [Theta_1, Theta_2, ..., Theta_l]` where `Theta_i` is the vector of parameters for layer `i`.
2. That said, our Fisher becomes `F = E[D(Theta)D(Theta)^T]`. This is an `l` by `l` block matrix, with the `(i, j)`-th block `F_i,j` given by `F_i,j = E[vec(DW_i)vec(DW_j)^T]`.
3. We can rewrite each block: `F_i,j = E[vec(DW_i)vec(DW_j)^T] = E[(a_i-1 x g_i)(a_j-1 x g_j)^T] = E[(a_i-1 x g_i)((a_j-1)^T x (g_j)^T] = E[((a_i-1)(a_j-1)^T) x (g_i)(g_j)^T]`. Here the sign `x` means [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).
4. Now, lets do initial approximation: `F_i,j = E[((a_i-1)(a_j-1)^T) x (g_i)(g_j)^T] approx= E[(a_i-1)(a_j-1)^T] x E[(g_i)(g_j)^T]`.
4. Approximating `F^-1` with block-diagonal.
5. Approximating `F^-1` with block-tridiagonal.
6. Estimating `E[a_i a_j^T]` and `E[g_i g_j^T]`.
7. Damping schemes (Levenberg-Marquardt, Tikhonov...)
