## Optimizing Neural Networks with Kronecker-factored Approximate Curvature

* Link: https://arxiv.org/abs/1503.05671

-----

The authors introduce an effective practical approximation of inverse of the Fisher information matrix. That allows to perform 
[Natural Gradient descent](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf) effectively.

K-FAC differs from other popular Hessian-Free optimization methods in a way that it does not involve running Conjugate Gradient
algorithm in order to compute `H^-1 * grad` (which is computationally expensive in practice). K-FAC can converge `>3.5x` faster
in `>14x` fewer iterations than SGD with Momentum 
(numbers are taken from tensorflow [implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/kfac) doc).
