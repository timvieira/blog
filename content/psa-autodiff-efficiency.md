title: Evaluating ∇f(x) is as fast as f(x)
date: 2016-09-25
comments: true
tags: math, calculus, autodiff, rant

Automatic differentiation ('autodiff' or 'backprop') is great&mdash;not just
because it makes it easy to rapidly prototype deep networks with plenty of
doodads and geegaws, but because it means that evaluating the gradient $\nabla
f(x)$ is as fast of computing $f(x)$. In fact, the gradient provably requires at
most a *small* constant factor more arithmetic operations than the function
itself.  Furthermore, autodiff tells us how to derive and implement the gradient
efficiently. This is a fascinating result that is perhaps not emphasized enough
in machine learning.

**The gradient should never be asymptotically slower than the function.** In my
recent [EMNLP'16 paper](/doc/2016-emnlp-vocrf.pdf), my coauthors and I found a
line of work on variable-order CRFs
([Ye+'09](https://papers.nips.cc/paper/3815-conditional-random-fields-with-high-order-features-for-sequence-labeling.pdf);
[Cuong+'14](http://www.jmlr.org/papers/volume15/cuong14a/cuong14a.pdf)), which
had an unnecessarily slow and complicated algorithm for computing gradients,
which was asymptotically (and practically) slower than their forward
algorithm. Without breaking a sweat, we derived a simpler and more efficient
gradient algorithm by simply applying backprop to the forward algorithm (and
made some other contributions).

**Many algorithms are just backprop.** For example, forward-backward and
inside-outside, are actually just instances of automatic differentiation (i.e.,
outside is just backprop on inside). This shouldn't be a surprise because these
algorithms are used to compute gradients. Basically, if you know backprop and
the inside algorithm, then you can derive the outside algorithm manually. I find
it easier to understand the outside algorithm via its connection to backprop,
then via
[the usual presentation](https://www.cs.jhu.edu/~jason/465/iobasics.pdf).

**Once you've grokked backprop, the world is your oyster!** You can backprop
through many approximate inference algorithms, e.g.,
[Stoyanov+'11](http://www.jmlr.org/proceedings/papers/v15/stoyanov11a/stoyanov11a.pdf)
and many of Justin Domke's papers, to avoid issues I've mentioned
[before](http://timvieira.github.io/blog/post/2015/02/05/conditional-random-fields-as-deep-learning-models/). You
can even backprop through optimization algorithms to get gradients of dev loss wrt
hyperparameters, e.g.,
[Domke'12](http://www.jmlr.org/proceedings/papers/v22/domke12/domke12.pdf) and
[Maclaurin+'15](https://arxiv.org/abs/1502.03492).

**There's at least one catch!** Although the *time* complexity of computing the
gradient is as good as the function, the *space* complexity may be much larger
because the autodiff recipe (at least the default reverse-mode one) requires memoizing
all intermediate quantities (e.g., the quantities you overwrite in a
loop). There are generic methods for balancing the time-space tradeoff in
autodiff, since you can (at least in theory) reconstruct the intermediate
quantities by playing the forward computation again from intermediate
checkpoints (at a cost to runtime, of course). A recent example is
[Gruslys+'16](https://arxiv.org/abs/1606.03401).

**A final remark**. Despite the name "automatic" differentiation, there is no
need to rely on software to "automatically" give you gradient routines. Applying
the backprop transformation is generally easy to do manually and sometimes more
efficient than using a library. Many autodiff libraries lack good support for
dynamic computation graph, i.e., when the structure depends on quantities that
vary with the input (e.g., sentence length).
