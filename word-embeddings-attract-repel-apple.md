## Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints

Paper: https://arxiv.org/abs/1706.00374
Code: https://github.com/nmrksic/attract-repel

-----

### Problem

Classic word embedding methods are powerful, but we want more. Especially, we often have knowledge databases (such as WordNet)
with relations between words, so we want to encode this knowledge in a distributional representation of words.

### Solution

Folks from Apple propose **Attract-Repel** algorithm that uses synonymy/antonymy constraints and obtains state-of-the-art
on word similarity tasks. This algorithm is a so called post-processor, it is applied to already learned
embeddings and finetunes them.

Also, they experiment with multi-lingual relations extracted from BabelNet and embed words from different languages into
the same semantic vector space.


### Details

* A batch of training examples consists of synonymic pairs of vectors and antonymic pairs. Two losses are defined for
both syn/ant pairs of words `x_l` and `x_r`.
* *Synonimic loss*: `S(x_l, x_r) = max(0, delta_syn + x_l*t_l - x_l*x_r) + max(0, delta_syn + x_r*t_r - x_l*x_r)`. Here `*`
is a dot product and `(t_l, t_r)` are negative samples. `t_l` is chosen from the remaining in-batch vectors so that `t_l` 
is the one closest to `x_l`. The same for `t_r` and `x_r`. So this loss tries to make `x_l` and `x_r` closer to each other than
to their negative examples.
* *Antonymic loss*: `A(x_l, x_r) = max(0, delta_ant + x_l*x_r - x_l*t_l) + max(0, delta_ant + x_l*x_r - x_r*t_r)`.
The loss tries to make `x_l` and `x_r` be further away from each other than from their negative examples.
* *Regularization*: `R(x, x_original) = lambda * ||x - x_original||^2`. This loss tries to make new vectors closer to
corresponding original vectors (after classic word embeddings training, e.g. word2vec).

### Experiments

* State-of-the-art on SimLex dataset (word similarity), both monolingual and multilingual.
* Training on pretrained vectors boosts a lot.

### What things could be improved

* How to encode more complex relations between words in this scheme? For example, relations on words as on entities
(e.g. relation `is a`: "king" is a "man"...)
