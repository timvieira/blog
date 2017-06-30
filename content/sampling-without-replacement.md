title: Sampling from a finite universe
date: 2017-06-27
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


**Setup**: Suppose you want to estimate an expectation of a derministic function
$f$ over a (large) finite universe of $n$ elements where each element $i$ has
probability $p_i$:

$$
\mu \overset{\tiny{\text{def}}}{=} \sum_{i=1}^n p_i f(i)
$$

However, $f$ is too expensive to evaluate $n$ times. Let's say that you have $m
\le n$ total evaluations to form your estimate.

Obviously, if we're happy evaluating $f$ a total of $n$ times, then we should
just compute $\mu$ exactly with the definition above.

**Why I'm writing this post**: Monte Carlo is often used in designing algorithms
  as a means to cheaply approximate intermediate expectations, think of
  stochastic gradient descent as a prime example. However, in many cases, we
  have a *finite* universe, i.e., we *could* enumerate all elements, but it's
  just inefficient to do so. In other words, sampling is merely a choice made by
  the algorithm designer, not a fundamental property of the environment, as it
  is typically in statistics. What can we do to improve estimation in this
  special setting? I won't get into bigger questions of how to design these
  algorithms, instead I'll focus on this specific type of estimation problem.


**An alternative formulation:** We can also formulate our estimation problem as
seeking a sparse, unbiased approximation to $\boldsymbol{p}$, the vector of
probabillties. We want our approximation, $\boldsymbol{s}$ to satisfy
$\mathbb{E}[\boldsymbol{s}] = \boldsymbol{p}$ and $\boldsymbol{s}$ while having
at most $m$ nonzeros entries (i.e., $\boldsymbol{s}$ is $m$-sparse). This will
suffice for estimating $\mu$ (above) because
$\mathbb{E}[\boldsymbol{s}^\top\! \boldsymbol{f}] =
\mathbb{E}[\boldsymbol{s}]^\top\! \boldsymbol{f} = \boldsymbol{p}^\top\!
\boldsymbol{f} = \mu$ where $\boldsymbol{f}$ is a vector of all $n$ values of
the function $f$. Obviously, you don't need to evaluate $f$ in places where
$\boldsymbol{s}$ is zero, thus this is a reasonable way to characterize the task
we described at the beginning which avoids evaluating $f$. Of course, unbiased
estimation of all probabillties is not *necessary* for unbiased estimation of
$\mu$ alone. However, this characterization is a good model for when we have
zero knowledge of $f$. This formulation might be of independent interest, since
a sparse, unbiased representation of a vector might be useful in some
applications.


**Monte Carlo:** The most well-known approach to this type of problem is Monte
Carlo (MC) estimation: sample $x^{(1)}, \ldots, x^{(m)}
\overset{\tiny\text{i.i.d.}}{\sim} p$, return $\widehat{\mu}_{\text{MC}} =
\frac{1}{m} \sum_{i = 1}^m f(x^{(i)})$.

*Remarks*: (1) Monte Carlo can be very inefficient because it resamples
high-probability items over and over again. (2) We can improve
efficiency&mdash;measured in $f$ evaluations&mdash;somewhat by caching past
evaluations of $f$. However, this introduces a serious *runtime*
inefficiency. (3) Even in our simple setting, MC never reaches *zero* error; it
only converges in an $\epsilon$-$\delta$ sense.

<!---
Remarks

 - We saw a similar problem where we kept sampling the same individuals over and
   over again in my
   [sqrt-biased sampling post](http://timvieira.github.io/blog/post/2016/06/28/sqrt-biased-sampling/).

 - My new nitpick (i.e., when I'm reviewer 2): don't say your samples go in a
   *set* unless you want duplicates to go away (or you know what you're
   doing). I've seen so many people make this little error. Just say the samples
   go in a bag, multiset, list, collection, etc. and probably stay away from curly
   braces.


**Drop the duplicates**: Speaking of putting samples in a set. It turns out that
it's sort of a good idea, if you do it correctly! Here something ridiculous that
is unbiased and has lower variance than Monte Carlo in our setting.

Take the same sample Monte Carlo sample from above, but throw out the duplicates
by putting them in a set!

$$
S = \textrm{set}(B)
$$

Now apply an importance-weight correction for having thrown things out. The
marginal probability that $i$ is in our sample is $q_i = 1 - (1 - p_i)^m$ (i.e.,
one minus the probability that we didn't sample $i$ in $m$ attempts).

$$
\widehat{\mu}_{\text{HT}} = \sum_{i \in S} \frac{p_i}{q_i} \cdot f(i)
$$


This estimator is really just kind of cute. I was surprised that it works better
than ordinary MC.

-->

**Sampling without replacement:** We can get around the problem of resampling
the same elements multiple times by simply *eliminating* them from consideration
after they have been sampled once. This is called a sampling *without
replacement* (SWOR) scheme. Note that there is no unique sampling without
replacement scheme; although, there does seem to be a de facto method. There are
lots of ways to do sample without replacement, e.g., any point process over the
universe will do as long as we can control the size.

Before getting into the details of sampling schemes, I'm going to talk about one
way that we can turn an SWOR scheme into an unbiased estimator.

**Horvitz-Thompson estimation:**
$$
\widehat{\mu}_{\text{HT}} = \sum_{i \in S} \frac{p_i}{q_i} \cdot f(i)
$$

where $S$ is a set of $m$ elements and $q_i$ is the marginal probability that
element $i$ appears in our sample $S$ (given our SWOR procedure).

This may look like importance sampling, but it is subtly different. In
particular, (1) lack of division by $1/m$ and (2) the fact there we don't have
duplicate elements. The justification for the HT estimator comes from the Best
Linear Unbiased Estimator (BLUE) framework, which is used to derive linear
estimators with minimal (optimal) variance properties.


#### Probability proportional sampling without replacement (PPSWOR)

This is the de facto sampling without replacement algorithm.

Here I'm using the Efraimidis and Spirakis algorithm, which is an efficient way
to do PPSWOR (even though it looks pretty dissimilar). I chose to include this
version because (1) it's a good way to implement PPSWOR, and (2) it's very
similar to the prioirty sampling (described later).

$$
\begin{align*}
&u_i, \ldots, u_n \overset{\tiny\text{i.i.d.}} \sim \mathcal{U}(0,1] \\
& k_i \leftarrow u_i^{1/p_i} \text{ for each $i$} \quad\color{grey}{\text{# random sort key }} \\
&S \leftarrow \{ \text{top-$m$ elements according to $k_i$} \} \\
\end{align*}
$$


Unfortunately, computing the inclusion probabilities ($q_i$) for this scheme is
"complicated" according to Duffield et al., (2005). Unfortunately, they do not
elaborate on the ways in which it is complicated, e.g., computation
complexity. I've tried work out an efficiently algorithm, but I'm not sure it's
possible (I have some inefficient ones).

Fear not! The next algorithm, priority sampling, is simpler and nearly
optimal. So we just forget PPSWOR ever happened :-P


#### Priority sampling (PS)

Here is pseudocode for priority sampling (Duffield et al., 2005). A remarkable
simple, elegant, and nearly optimal algorithm for our task.

$$
\begin{align*}
&u_i, \ldots, u_n \overset{\tiny\text{i.i.d.}} \sim \mathcal{U}(0,1] \\
& k_i \leftarrow p_i/u_i \text{ for each $i$} \quad\color{grey}{\text{# random sort key }} \\
&S \leftarrow \{ \text{top-$m$ elements according to $k_i$} \} \\
&\tau \leftarrow (m+1)^{\text{th}}\text{ largest }k_i \\
&\widehat{\mu}_{\text{PS}} = \sum_{i \in S} \max\left( p_i, \tau \right) \cdot f(i)
\end{align*}
$$


**Properties**:

 - The procedure works as a reservoir sampling scheme, since the keys and
   threshold can be computed as we run and stopped at any time, in principle.

 - Priority sampling is a near-optimal m-sparse estimator. The proof
   [(Szegedy, 2005)](
   https://www.cs.rutgers.edu/~szegedy/PUBLICATIONS/full1.pdf) seems to be a bit
   involved. So I'll only mention the main theorem, which says that the variance
   of priority sampling with $m$ samples is no worse than the best possible
   $(m-1)$-sparse estimator in terms of variance, measured as
   $\textrm{Var}(\sum_i w_i)$, which notably ignores $f$.

 - Samples are uncorrelated, i.e., $\textrm{Cov}[s_i, s_j] = 0$ for $i \ne j$
   and $m \ge 2$.


**Remarks**:

 - Note that the inclusion probabilities of PS are quite simple,
   $\frac{p_i}{q_i} = \max\left( p_i, \tau \right)$.



## Experiments

You can get the code for replicating these experiments in this gist.

The improvement of PS over MC is shocking, check it out:


## Further reading

If you liked this post, you might like my other
[posts tagged with sampling](http://timvieira.github.io/blog/tag/sampling.html).
