title: Conditional random fields as Deep learning models?
date: 2015-02-05
comments: true
tags: machine-learning, deep-learning, structured-prediction

This post is intended to convince conditional random field (CRF) lovers that
deep learning might not be as crazy as it seems. And maybe even convince some
deep learning lovers that the graphical models might have interesting things to
offer.

In the world of structured prediction, we are plagued by the high-treewidth
problem -- models with loopy factors are "bad" because exact inference is
intractable. There are three common approaches for dealing with this problem:

1. Limit the expressiveness of the model (i.e., don't use to model you want)

2. Change the training objective

3. Approximate inference

Approximate inference is tricky. Things can easily go awry.

For example, structured perceptron training with loopy max-product BP instead of
exact max product can diverge
[(Kulesza & Pereira, 2007)](http://papers.nips.cc/paper/3162-structured-learning-with-approximate-inference.pdf). Another
example: using approximate marginals from sum-product loopy BP in place of the
true marginals in the gradient of the log-likelihood. This results in a
different nonconvex objective function. (Note:
[sometimes](http://aclweb.org/anthology/C/C12/C12-1122.pdf) these loopy BP
approximations works fine.)

It looks like using approximate inference during training changes the training
objective.

So, here's a simple idea: learn a model which makes accurate predictions given
the approximate inference algorithm that will be used at test-time. Furthermore,
we should minimize empirical risk instead of log-likelihood because it is robust
to model miss-specification and approximate inference. In other words, make
training conditions as close as possible to test-time conditions.

Now, as long as everything is differentiable, you can apply automatic
differentiation (backprop) to train the end-to-end system. This idea appears in
a few publications, including a handful of papers by Justin Domke, and a few by
Stoyanov & Eisner.

Unsuprisingly, it works pretty well.

I first saw this idea in Stoyanov & Eisner (2011). They use loopy belief
propagation as their approximate inference algorithm. At the end of the day,
their model is essentially a deep recurrent network, which came from unrolling
inference in a graphical model. This idea really struck me because it's clearly
right in the middle between graphical models and deep learning.

You can immediately imagine swapping in other approximate inference algorithms
in place of loopy BP.

Deep learning approaches get a bad reputation because there are a lot of
"tricks" to get nonconvex optimization to work and because model structures are
more open ended. Unlike graphical models, deep learning models have more
variation in model structures. Maybe being more open minded about model
structures is a good thing. We seem to have hit a brick wall with
likelihood-based training. At the same time, maybe we can port over some of the
good work on approximate inference as deep architectures.
