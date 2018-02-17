## Continuous control with deep reinforcement learning (DDPG)

* Paper: https://arxiv.org/abs/1509.02971

-----

### Problem & Idea

D. Silver & co. introduced DPG ([my notes](https://github.com/persiyanov/just-paper-notes/blob/master/dpg.md)). After DQN has succeded on Atari games it is the time for trying DPG with deep neural nets, evaluating it on MuJoCo environments. 

This paper is practical, there is no new theory introduced (but, anyway, it provides the great result).

### Details

* DPG doesn't work with deep nets if applied straightforwardly.
* Several hacks have been adopted from [DQN paper](https://arxiv.org/abs/1312.5602): Target networks, Experience replay buffer. Also, batch normalization is used for faster training.
* Target networks are used for both for critic and actor (then, we generate targets from them in order to train our critic).
* Train policy with DPG update (here we do not use target nets, but learn on samples from behaviour policy which is actually
noisy version of our current policy)

#### DDPG algorithm pseudocode

![](https://image.prntscr.com/image/kPRlslstTSSWwENsCSEFUQ.png)

### Experiments

On several MuJoCo environments with continuous action spaces. The results show that all hacks contributes into final performance in a good way. Without them (standard DPG) performs worse.
