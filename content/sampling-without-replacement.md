title: Sampling from a finite universe
date: 2017-06-30
comments: true
status: draft
tags: sampling, statistics, reservoir-sampling


<style>
.toggle-button {
    background-color: #555555;
    border: none;
    color: white;
    padding: 10px 15px;
    border-radius: 6px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    cursor: pointer;
}
.derivation {
  background-color: #f2f2f2;
  border: thin solid #ddd;
  padding: 10px;
  margin-bottom: 10px;
}
</style>
<script>
// workaround for when markdown/mathjax gets confused by the
// javascript dollar function.
function toggle(x) { $(x).toggle(); }
</script>


**Setup**: Suppose we want to estimate an expectation of a derministic function
$f$ over a (large) finite universe of $n$ elements where each element $i$ has
probability $p_i$:

$$
\mu \overset{\tiny{\text{def}}}{=} \sum_{i=1}^n p_i f(i)
$$

However, $f$ is too expensive to evaluate $n$ times. So let's say that we have
$m \le n$ evaluations to form our estimate. (Obviously, if we're happy
evaluating $f$ a total of $n$ times, then we should just compute $\mu$ exactly
with the definition above.)

<!--
**Why I'm writing this post**: Monte Carlo is often used in designing algorithms
as a means to cheaply approximate intermediate expectations, think of stochastic
gradient descent as a prime example. However, in many cases, we have a *finite*
universe, i.e., we *could* enumerate all elements, but it's just inefficient to
do so. In other words, sampling is merely a choice made by the algorithm
designer, not a fundamental property of the environment, as it is typically in
statistics. What can we do to improve estimation in this special setting? I
won't get into bigger questions of how to design these algorithms, instead I'll
focus on this specific type of estimation problem.
-->

**Monte Carlo:** The most well-known approach to this type of problem is Monte
Carlo (MC) estimation: sample $x^{(1)}, \ldots, x^{(m)}
\overset{\tiny\text{i.i.d.}}{\sim} p$, return $\widehat{\mu}_{\text{MC}} =
\frac{1}{m} \sum_{i = 1}^m f(x^{(i)})$. *Remarks*: (1) Monte Carlo can be very
inefficient because it resamples high-probability items over and over again. (2)
We can improve efficiency&mdash;measured in $f$ evaluations&mdash;somewhat by
caching past evaluations of $f$. However, this introduces a serious *runtime*
inefficiency and requires modifying the method to account for the fact that $m$
is not fixed ahead of time. (3) Even in our simple setting, MC never reaches
*zero* error; it only converges in an $\epsilon$-$\delta$ sense.

<!---
Remarks

 - We saw a similar problem where we kept sampling the same individuals over and
   over again in my
   [sqrt-biased sampling post](http://timvieira.github.io/blog/post/2016/06/28/sqrt-biased-sampling/).
-->

**Sampling without replacement:** We can get around the problem of resampling
the same elements multiple times by sampling $m$ distinct elements. This is
called a sampling *without replacement* (SWOR) scheme. Note that there is no
unique sampling without replacement scheme; although, there does seem to be a
*de facto* method (PPSWOR; which we won't be using in this post!). There are
lots of ways to do sample without replacement, e.g., any point process over the
universe will do as long as we can control the size.

**An alternative formulation:** We can also formulate our estimation problem as
seeking a sparse, unbiased approximation to a vector $\boldsymbol{x}$. We want
our approximation, $\boldsymbol{s}$ to satisfy $\mathbb{E}[\boldsymbol{s}] =
\boldsymbol{x}$ and while $|| \boldsymbol{s} ||_0 \le m$. This will suffice for
estimating $\mu$ (above) when $\boldsymbol{x}=\boldsymbol{p}$, the vector of
probabillties, because $\mathbb{E}[\boldsymbol{s}^\top\! \boldsymbol{f}] =
\mathbb{E}[\boldsymbol{s}]^\top\! \boldsymbol{f} = \boldsymbol{p}^\top\!
\boldsymbol{f} = \mu$ where $\boldsymbol{f}$ is a vector of all $n$ values of
the function $f$. Obviously, we don't need to evaluate $f$ in places where
$\boldsymbol{s}$ is zero so it works for our budgeted estimation task. Of
course, unbiased estimation of all probabillties is not *necessary* for unbiased
estimation of $\mu$ alone. However, this characterization is a good model for
when we have zero knowledge of $f$. Additionally, this formulation might be of
independent interest, since a sparse, unbiased representation of a vector might
be useful in some applications (e.g., replacing a dense vector with a sparse
vector can lead to more efficient computations).

**Priority sampling**: Priority sampling (Duffield et al., 2005;
[Duffield et al., 2007](http://nickduffield.net/download/papers/priority.pdf))
is a remarkable simple algorithm, which is essentially optimal for our task if
we assume no knowledge of $f$. Here is pseudocode for priority sampling (PS),
based on the alternative formulation.

$$
\begin{align*}
&\textbf{inputs: } \text{vector } \boldsymbol{x} \in \mathbb{R}^n, \text{budget } m \in \{1, \ldots, n\}\\
&u_i, \ldots, u_n \overset{\tiny\text{i.i.d.}} \sim \textrm{Uniform}(0,1] \\
& k_i \leftarrow x_i/u_i \text{ for each $i$} \quad\color{grey}{\text{# random sort key }} \\
&S \leftarrow \{ \text{top-$m$ elements according to $k_i$} \} \\
&\tau \leftarrow (m+1)^{\text{th}}\text{ largest }k_i \\
& s_i = \begin{cases}
  \max\left( x_i, \tau \right)  & \text{ if } i \in S \\
  0                             & \text{ otherwise}
\end{cases} \\
&\textbf{return }\boldsymbol{s}
\end{align*}
$$

For estimating $\mu$, use the following in place of the last line:
$$
\widehat{\mu}_{\text{PS}} = \sum_{i \in S} \max\left( x_i, \tau \right) \cdot f(i)
$$

**Properties**:

 - It satisfies our task criteria: it is both unbiased and sparse (i.e., under
   the evaluation budget).

 - Can be modified slightly to support streaming $x_i$, since the keys and
   threshold can be computed as we run, which means it can be stopped at any
   time, in principle.

 - Priority sampling is a near-optimal $m$-sparse estimator for subset sums
   [(Szegedy, 2005)](https://www.cs.rutgers.edu/~szegedy/PUBLICATIONS/full1.pdf),
   i.e., estimating $\sum_{i=1}^n x_i$. The variance of priority sampling with
   $m$ samples is no worse than the best possible $(m-1)$-sparse estimator in
   terms of variance. Of course, if we have some knowledge about $f$, we can
   beat PS (e.g.,. via
   [importance sampling](http://timvieira.github.io/blog/post/2016/05/28/the-optimal-proposal-distribution-is-not-p/)
   or by modifying PS to sample proportional to $x_i = p_i \!\cdot\! |f_i|$ (as
   well as other straightforward modifications), but presumably with a surrogate
   for $f_i$ because we don't want to evaluate it).

 - Samples are uncorrelated, i.e., $\textrm{Cov}[s_i, s_j] = 0$ for $i \ne j$
   and $m \ge 2$. This is surprising since $s_i$ and $s_j$ are related via the
   threshold $\tau$.


## Experiments

You can get the Jupyter notebook for replicating these experiments
[here](https://github.com/timvieira/blog/blob/master/content/notebook/Priority%20Sampling.ipynb).
So download the notebook and play with it!

The improvement of priority sampling (PS) over Monte Carlo (MC) is pretty nice,
check it out:

<center>
![Priority sampling vs. Monte Carlo](http://timvieira.github.io/blog/images/ps-mc.png)
</center>

The shaded region indicates the 10% and 90% percentiles over 20,000
replications, which gives a sense of the variability of each estimator. The
x-axis is the sampling budget, $m \le n$.

The plot shows a small example with $n=50$. We see that PS's variability
actually goes to zero, unlike Monte Carlo, which is still pretty inaccurate even
at $m=n$. (Note that MC's x-axis measures raw evaluations, not distinct ones.)



## Further reading

If you liked this post, you might like my other posts tagged with
[sampling](http://timvieira.github.io/blog/tag/sampling.html) or
[reservoir sampling](http://timvieira.github.io/blog/tag/reservoir-sampling.html).
