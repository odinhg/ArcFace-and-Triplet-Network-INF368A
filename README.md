# INF368A Exercise 2
**Odin Hoff Gardå**

![Plankton](figs/plankton.png)

## Todo

- Create triplet data loader
- Implement triplet loss
- Could do with some better config system allowing multiple models to be compared easily
- Remove files no longer needed from exercise 1


## Task 1
To train the triplet loss model, run `train.py` and choose the `triplet_loss` config when prompted.

### Triplet Loss
I choose to implement a triplet loss architecture with online triplet mining. The loss function is defined as
$$
\mathcal(L)(x_a, x_p, x_n) = \operatorname{ReLU}(\Vert f(x_a) - f(x_p) \Vert^2 - \Vert f(x_a) - f(x_n) \Vert^2 + m)
$$
where $m$ is a margin parameter and $(x_a, x_p, x_n)$ is a valid triplet. That is, $C(x_a) = C(x_p) \neq C(x_n)$ and $x_a \neq x_p$ where $C(x)$ denotes the class of $x$.

### Mining strategy
I implemented an online triplet mining strategy. That is, we extract the embeddings for each sample in the mini-batch, compute the euclidean distance in the embedding space and pick the most useful triplets based on our mining policy. The reason for not doing all triplets (batch-all) or random triplets is to prevent adding triplets with zero loss as these are too easy to learn from. There are two different policies for positives and negatives implemented, respectively. These are:

**Easy positive:** Given an anchor $x_a$, we choose the positive $x_p$ minimizing $\Vert f(x_a) - f(x_p)\Vert^2$.
**Hard positive:** Given an anchor $x_a$, we choose the positive $x_p$ maximizing $\Vert f(x_a) - f(x_p)\Vert^2$.
**Semi-hard negative:** We pick a negative $x_n$ satisfying $\Vert f(x_a) - f(x_p) \Vert^2 < \Vert f(x_a) - f(x_n) \Vert^2 < \Vert f(x_a) - f(x_p) \Vert^2 + m$.
**Hard negative:** We pick a negative $x_n$ satisfying $\Vert f(x_a) - f(x_p) \Vert^2 > \Vert f(x_a) - f(x_n)$. 

If we do not find any negatives satisfying the active policy, the one minimizing $\Vert f(x_a) - f(x_n)\Vert^2$ is choosen to prevent NaNs.

Without these mining policies, the model collapsed. That is, it was satisfied with learning $f(x)=0$ as the loss converged to the value of $m$. To prevent collapse, we train the model using different combinations of the above policies:

1. Epoch 0 to 9: Easy positives and semi-hard negatives.
2. Epoch 10 to 14: Easy positives and hard negatives.
3. Epoch 15 to 24: Hard positives and hard negatives.

The use of easy positives in training was inspired by the paper [Improved Embeddings with Easy Positive Triplet Mining](arxiv.org/abs/1904.04370).

## Task 2
- Same backbone, but implement CosFace, ArcFace or somthing like that.
- Train model and report

## Task 3 and 4
Embed data and report average distances (use appropriate distance measure)

## Task 5
Visualise representations using UMAP for both models

## Task 6
Display images close to center, far-away from center and closest images from other classes. For both models.

## Task 7
Train classifiers on embeddings only.
