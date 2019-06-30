Title: Gumbel-max trick and weighted reservoir sampling
date: 2014-08-01
comments: true
tags: sampling, Gumbel, reservoir-sampling, sampling-without-replacement

A while back, [Hanna](http://people.cs.umass.edu/~wallach/) and I stumbled upon
the following blog post:
[Algorithms Every Data Scientist Should Know: Reservoir Sampling](http://blog.cloudera.com/blog/2013/04/hadoop-stratified-randosampling-algorithm),
which got us excited about reservoir sampling.

Around the same time, I attended a talk by
[Tamir Hazan](http://cs.haifa.ac.il/~tamir/) about some of his work on
perturb-and-MAP
[(Hazan & Jaakkola, 2012)](http://cs.haifa.ac.il/~tamir/papers/mean-width-icml12.pdf),
which is inspired by the
[Gumbel-max-trick](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/)
(see [previous post](/blog/post/2014/07/31/gumbel-max-trick/)). The apparent similarity between weighted reservoir sampling and the Gumbel-max trick lead us to make some cute connections, which I'll describe in this post.

**The problem**: We're given a stream of unnormalized probabilities, $x_1, x_2, \cdots$. At any point in time $t$ we'd like to have a sampled index $i$ available, where the probability of $i$ is given by $\pi_t(i) = \frac{x_i}{
\sum_{j=1}^t x_j}$.

Assume, without loss of generality, that $x_i > 0$ for all $i$. (If any element has a zero weight we can safely ignore it since it should never be sampled.)

**Streaming Gumbel-max sampler**: I came up with the following algorithm, which is a simple "modification" of the Gumbel-max-trick for handling streaming data:

$a = -\infty; b = \text{null}  \ \ \text{# maximum value and index}$
for $i=1,2,\cdots;$ do:
:  \# Compute log-unnormalized probabilities
:  $w_i = \log(x_i)$
:  \# Additively perturb each weight by a Gumbel random variate
:  $z_i \sim \text{Gumbel}(0,1)$
:  $k_i = w_i + z_i$
:  \# Keep around the largest $k_i$ (i.e. the argmax)
:  if $k_i > a$:
:  $\ \ \ \ a = k_i$
:  $\ \ \ \ b = i$

If we interrupt this algorithm at any point, we have a sample $b$.

After convincing myself this algorithm was correct, I sat down to try to
understand the algorithm in the blog post, which is due to Efraimidis and
Spirakis (2005) ([paywall](http://dl.acm.org/citation.cfm?id=1138834),
[free summary](http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf)). They
looked similar in many ways but used different sorting keys/perturbations.

**Efraimidis and Spirakis (2005)**: Here is the ES algorithm for weighted
reservoir sampling

$a = -\infty; b = \text{null}$
for $i=1,2,\cdots;$ do:
:  \# compute randomized key
:  $u_i \sim \text{Uniform}(0,1)$
:  $e_i = u_i^{(\frac{1}{x_i})}$
:  \# Keep around the largest $e_i$
:  if $k_i > a$:
:  $\ \ \ \ a = k_i$
:  $\ \ \ \ b = i$

Again, if interrupt this algorithm at any point, we have our sample $b$. Note
that you can simplify $e_i$ so that you don't have to compute pow (which is nice
because pow is pretty slow). It's equivalent to use $e'_i = \log(e_i) =
\log(u_i)/x_i$ because $\log$ is monotonic. (Note that $-e'_i \sim
\textrm{Exponential}(x_i)$.)

<!--
I find this version of the algorithm more intuitive, since it's well-known that
$\left(\underset{{i=1 \ldots t}}{\min} \textrm{Exponential}(x_i) \right) =
\textrm{Exponential}\left(\sum_{i=1}^t x_i \right)$. This version makes it clear
that minimizing is actually summing. However, we want the argmin, which is
distributed according to $\pi_t$.
-->

**Relationship**: Let's try to relate these algorithms. At a high level, both
algorithms compute a randomized key and take an argmax. What's the relationship
between the keys?

First, note that a $\text{Gumbel}(0,1)$ variate can be generated via
$-\log(-\log(\text{Uniform}(0,1)))$. This is a straightforward application of
the
[inverse transform sampling](http://en.wikipedia.org/wiki/Inverse_transform_sampling)
method for random number generation. This means that if we use the same sequence
of uniform random variates then, $z_i = -\log(-\log(u_i))$.

However, this does not give use equality between $k_i$ and $e_i$, but it does
turn out that $k_i = -\log(-\log(e_i))$, which is useful because this is a
monotonic transformation on the interval $(0,1)$. Since monotonic
transformations preserve ordering, the sequences $k$ and $e$ result in the same
comparison decisions, as well as, the same argmax. In summary, the algorithms
are the same!

**Extensions**: After reading a little further along in the ES paper, we see
that the same algorithm can be used to perform *sampling without replacement* by
sorting and taking the elements with the highest keys. This same modification applies to the Gumbel-max-trick because the keys have precisely the same ordering as ES. In practice, we don't sort the key, but instead, use a bounded priority queue.

**Closing**: To the best of my knowledge, the connection between the
Gumbel-max trick and ES is undocumented. Furthermore, the Gumbel-max-trick is not known as a streaming algorithm, much less known to perform sampling without replacement! If you know of a reference, let me know.


**How to cite this article**: If you found this article useful, please cite it as

<pre style="background-color: white; color: black; border: #333;">
@misc{vieira2014gumbel,
    title = {Gumbel-max trick and weighted reservoir sampling},
    author = {Tim Vieira},
    url = {http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/},
    year = {2014}
}
</pre>

## Further reading

I have a few other articles that are related to
[sampling without replacement](http://timvieira.github.io/blog/tag/sampling-without-replacement.html),
[reservoir sampling](http://timvieira.github.io/blog/tag/reservoir-sampling.html),
and [Gumbel tricks](http://timvieira.github.io/blog/tag/gumbel.html).

Here are a few interesting papers that build on the ideas in this article.

- Wouter Kool, Herke van Hoof, and Max Welling. 2019.
[Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement](https://arxiv.org/abs/1903.06059)

- Sang Michael Xie and Stefano Ermon. 2019. [Reparameterizable Subset Sampling via Continuous Relaxations](https://arxiv.org/abs/1901.10517)
