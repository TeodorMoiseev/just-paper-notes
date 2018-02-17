## Continuous control with deep reinforcement learning (DDPG)

* Paper: https://arxiv.org/abs/1509.02971

-----

[TODO!]
* Introduced DPG. Let's apply it to deep neural nets!
* It doesn't work if applied straightforwardly. Need to use exp replay, target nets (they are hacks for stable learning).
* Standard Q-learning can't be applied to continuous action spaces.
* Target nets both for critic and actor (then, we generate targets from them in order to train our critic)
* Train policy with DPG update (here we do not use target nets, but learn on samples from behaviour policy which is actually
noisy version of our current policy)


### Problem & Idea


### Details


### Experiments


### What things could be improved
