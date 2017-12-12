## Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation 

Link: https://arxiv.org/abs/1604.06057

-----

### Problem

In many environments the agent is unable to learn robust, efficient policy due to
very complex environment and, in result, **inefficient exploration**.

### Solution

Hierarchical Reinforcement Learning (HRL) proposes framework where there are two level of policies -- high-level policy,
which chooses the subgoal, and low-level policy, which acts within the subgoal and achieves it.

**This paper proposes** h-DQN, which allows flexible goal specifications for classical DQN.
In result, good exploration can be achieved.


### Details

* In order to obtain good exploration, the authors use a notion of goals `g \in G`, which provide
intrinsic motivation for the agent.
* The agent focuses on setting and achieving sequences of goals in order to maximize a sum of extrinsic rewards.
* Use two level hierarchy. **Top-level** neural network (*meta-controller*) takes in the state `s` and produces a new goal `g`.
**Low-level** network (*controller*) uses both current `s` and `g` and produces actions until the goal is reached
or episode is ended.
* Instead of learning ordinary `Q(s, a)`, they learn two Q-functions for each level in the hierarchy.
**Top-level** Q-function is `Q(s, g)` like original DQN. **Low-level** Q-function is `Q(s, a; g)`.
* Use two experience replays accordingly, on different time-scales each. 
**Top-level** exp replay stores transitions `(s, g, F, s')` where `F` is a total extrinsic (from environment)
reward which was obtained while completing goal `g`. **Low-level** exp replay stores transitions `({s, g}, a, r, {s', g})`
where `r` is an intrinsic reward for one time-step (in experiments, they use positive reward iff `s'` is final for the goal `g`.


### Experiments

* Solve **ATARI Montezuma’s Revenge** game. They use handcrafted goals, based on distance between 
the agent and particular object on the screen.


### What things could be improved

* Goals are needed to be handcrafted (along with internal critic which produces intrinsic reward withing the goal).
