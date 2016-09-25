title: Evaluating ∇f(x) is as fast as f(x)
date: 2016-09-25
comments: true
status: draft
tags: math, calculus, autodiff, rant

Automatic differentiation (autodiff) is great. Not just because it makes it easy
to slap together some wacky deep network with plenty of doodads and geegaws, but
because it means that evaluating the gradient, $\nabla f(x)$ is as fast of
computing $f(x)$ (provably within a small constant factor). This is a
fascinating result that is not emphasized enough in machine learning and
theoretical computer science education and perhaps not even known to some.

In my recent [EMNLP'16 paper](/doc/2016-emnlp-vocrf.pdf), my coauthors and I
found a line of work on variable-order CRFs
([Ye+'09](https://papers.nips.cc/paper/3815-conditional-random-fields-with-high-order-features-for-sequence-labeling.pdf);
[Cuong+'14](http://www.jmlr.org/papers/volume15/cuong14a/cuong14a.pdf)), which
had an unnecessarily slow algorithm for computing gradients, which was
asymptotically (and practically) slower than the forward algorithm---we fixed
it, of course (and made some other contributions).

Another thing that seems to surprise people is that many well-known algorithms,
such as, forward-backward and inside-outside, are actually just instances of
automatic differentiation (i.e., outside is just autodiff on inside). This
shouldn't be a surprise because these are used to compute gradients... (Of
course, inside-outside can compute more than gradients because it's a semiring
algorithm, e.g., [Li & Eisner,'09](http://www.aclweb.org/anthology/D09-1005).)

There is one catch! Although the *time* complexity is the same, the *space*
complexity may be much larger because the default autodiff recipe requires
memoizing all intermediate quantities (e.g., the quantities you overwrite in a
loop). There are generic methods for balancing the time-space tradeoff in
autodiff, since you can (at least in theory) reconstruct the intermediate
quantities by playing the forward computation again from intermediate
checkpoints (at a cost to runtime, of course).

As a final remark, despite the name "automatic" differentiation, there is no
need to rely on software to "automatically" give you gradient routines. Applying
the autodiff transformation is generally easy to do manually and sometimes more
efficient than using a library. Most autodiff libraries stink at computing
things where the computation graph depends on quantities that vary with the
input (e.g., sentence length).
