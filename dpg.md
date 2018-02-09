## Deterministic Policy Gradient Algorithms

* Paper: http://proceedings.mlr.press/v32/silver14.pdf
* Supplementary material: http://proceedings.mlr.press/v32/silver14-supp.pdf

-----

### Problem & Idea

The classic _policy gradient theorem_ is formulated for stochastic policies ![](http://mathurl.com/ybsnhmwz.png) and gives an
expression for policy performance gradient ![](http://mathurl.com/y8ltt3xt.png).

**DPG paper** introduces the similar result but for deterministic policies of form ![](http://mathurl.com/yanduehc.png). Why
does it useful in practice? The answer is that DPG integrates only over state space. In contrast, stochastic policy gradient
integrates both over action and state spaces and leads to higher sample complexity (or worse approximation of the gradient).

In order to continue to explore the environment the authors introduce off-policy actor-critic algorithm which learns about
deterministic policy but uses stochastic behaviour policy.

### Details

1. In discrete action spaces and deterministic policies, value-based methods could be applied, e.g. Q-learning.
In that case, on the `k`-th iteration we could use the following policy improvement step ![](http://mathurl.com/y7qgo6au.png).

2. Unfortunately, in continuous action space this optimization becomes intractable. Instead, an attractive alternative is to
move the policy in the direction of the gradient of ![](http://mathurl.com/lzhndgq.png). Of course, in that case, we need to
use parametrized policy. Then, update is ![](http://mathurl.com/ydd7fwzl.png).

3. Let's use the chain rule for ![](http://mathurl.com/lzhndgq.png). Then we get ![](http://mathurl.com/y8owce2r.png).
Here, ![](http://mathurl.com/ybe4bkah.png) is a Jacobian matrix with `dim(theta)` rows and `dim(a)` columns.

4. But that's the intuition! What about the theorem? It's not obvious, but that update is correct.
This is formulated in *policy gradient theorem for deterministic policies*. 
Now, the objective is defined like this ![](http://mathurl.com/y9g2m99k.png) and the theorem states that:

![](http://mathurl.com/ydxu5n88.png).

### Experiments

### What things could be improved
