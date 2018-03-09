## Q-Prop: Sample-Eficient Policy Gradient With an Off-Policy Critic

* Paper: https://arxiv.org/abs/1611.02247

-----

### Problem & Idea

1. We want to do sample efficient RL, because there are a lot of environments, where sample collection is long and expensive.
2. Monte-Carlo policy gradient methods provide stable learning, but at cost of high variance. Very sample intensive.
3. TD-style methods (Off-PAC, Q-learning) are more sample-efficient (because they are off-policy) but biased.
4. Authors combined these two approaches and presented Q-Prop algorithm. It can reduce variance without adding bias.

**Core idea**:
* Q-Prop learns `Q(s, a)` function in off-policy mode.
* It uses the first-order Taylor expansion of the critic as a control variate.
* The method can be seen as using the off-policy critic to reduce variance in policy gradient or using on-policy Monte Carlo returns to correct for bias in the critic gradient.

### Details


### Experiments
