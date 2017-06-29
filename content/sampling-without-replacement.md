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
zero knowledge of $f$.


### Monte Carlo (MC)

The most well-known approach to this type of problem is Monte Carlo estimation.

$$
\begin{align*}
&x_1, \ldots, x_m \cdots \overset{\tiny\text{i.i.d.}}{\sim} p \\
&B \leftarrow [x_1, \ldots, x_m] \\
&\widehat{\mu}_{\text{MC}} = \frac{1}{m} \sum_{j \in B} f(j)
\end{align*}
$$


**Remarks**:

 - Monte Carlo can be very inefficient because it resamples high-probability
   items over and over again.

 - We can improve efficiency&mdash;measured in $f$ evaluations&mdash;somewhat by
   caching past evaluations of $f$. However, this introduces a *runtime*
   inefficiency, which is undesirable.

 - This estimator never reaches zero error, it only converges in an
   $\epsilon$-$\delta$ sense.

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


### Sampling without replacement (SWOR)

We can get around the problem of resampling the same elements multiple times by
simply eliminating them from consideration after they have been sampled
once. This is called a sampling *without replacement* scheme.

Note that there is no "one" sampling without replacement scheme. There are lots
of ways to do sample without replacement, e.g., any point process over the
universe will do as long as we can control the size.



### Horvitz-Thompson estimator

In order to accommodate a sampling without replacement scheme, we'll use an
estimator with the following form, known as the Horvitz-Thompson estimator:

$$
\widehat{\mu}_{\text{HT}} = \sum_{i \in S} \frac{p_i}{q_i} \cdot f(i)
$$

where $q_i$ is the probability that element $i$ appears in our sample of size
$m$.

This may look like importance sampling, but it is subtly different. In
particular, note the lack of division by $1/m$ and the fact there we don't have
duplicate sampled elements. The justification for the HT estimator comes from
the Best Linear Unbiased Estimator (BLUE) framework, which is used to derive
linear estimators with minimal (optimal) variance properties.


### What SWOR scheme should we use?

#### Probability proportional sampling without replacement (PPSWOR)

This is the de facto sampling without replacement algorithm.

Unfortunately, computing the inclusion probabilities ($q_i$) for this scheme is
"complicated" according to Duffield et al., (2005). However, do not elaborate on
the ways in which it is complicated, e.g., computation complexity. I've tried
work out an efficiently algorithm, but I'm not sure it's possible (I have some
inefficient ones).

Fear not! The next algorithm, priority sampling, is simpler and nearly
optimal. So we just forget PPSWOR ever happened :-P


#### Priority sampling (PS)

Here is pseudocode for priority sampling (Duffield et al., 2005). A remarkable
simple, elegant, and nearly optimal algorithm for our task.

$$
\begin{align*}
&u_i, \ldots, u_n \overset{\tiny\text{i.i.d.}} \sim \mathcal{U}(0,1] \\
& \text{% Define a random sort key for each element} \\
& k_i \leftarrow p_i/u_i \text{ for each $i$}\\
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
   $(m-1)$-sparse estimator (in terms of variance).

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



## Optimal variance estimator

The optimal variance estimator with an *expected* sample size is threshold
sampling (PS's sibling) which&mdash;like priority sampling is based on the same
thresholding ($\tau$) idea&mdash;but $\tau$ is set prior to sampling (in a way
that guarantees a particular expected sample size).

If the entire stream is known, then $\tau$ can be computed such that the
expected size is $m$, but this doesn't guarantee exactly $m$, in order to
operate in the reservoir setting, we have to guess a value for $\tau$, which
could be arbitrarily off.

The intuition for $(m-1)$ optimality is that priority sampling is a lot like
threshold sampling, but where estimating $\tau$ adds exactly one sample's worth
of variability.

## Bonus: Comparsion to Gumbel-max trick and others

The key and overall method bears some resemblance to PPSWOR sampling via the
Gumbel-max trick [LINK]&mdash;or equivalently weighted reservoir sampling
[CITE]&mdash;
([see my post](http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/)). However,
it is different. As I noted, computing inclusion probabilities for the PPSWOR
replacement scheme is apparently "complicated." But, let's humor the comparison

The key for ES sampling is
$$
e_i = u_i^{1/p_i}
$$

If we apply the order-preserving transformation, $-\log(-\log(x))$, we get the GM
trick,
$$
g_i = -\log(-\log(e_i)) = -\log(-\log(u_i^{1/p_i})) = \log(p_i) -\log(-\log(u_i))
$$

(It turns out that $-\log(-\log(u_i))$ is a $\textrm{Gumbel}(0,1)$ random
variate, hence the name of the trick.)

The key for PS is
$$
k_i = p_i/u_i
$$

If we take a $\log$ of the key, we preserve order
$$
\log k_i = \log(p_i/u_i) = \log(p_i) - \log(u_i)
$$

This is kind of interesting because we're only a $-\log$ away from GM's
key. This beg's the question what is $-\log(u_i)$. It turns out this is an
$\textrm{Exp}(1)$ R.V.

## PS's sampling distribution is pretty different

Duffield et al (2005) give a nice example that illustrates how PS has a
different sampling probability than most schemes (e.g., PPSWOR).

Consider $n=2$ and $m=1$. Further, suppose $w_1 < w_2$.


We can contrast this with other methods (e.g., MC or PPSWOR), that will pick
$i=1$ with probability, strictly higher probability,
$$
p(i=1 | \text{OTHER}) = \frac{w_1}{w_1 + w_2} > \frac{w_1}{2\cdot w_2}.
$$

(The inequality holds because $w_1 < w_2$.)

<button class="toggle-button" onclick="toggle('#ps-sampling-prob');">Derivation</button>
<div id="ps-sampling-prob" class="derivation" style="display: block;">

Now, let's work out the probability that we sample the lower-weight element
($i=1$).

$$
\begin{eqnarray*}
p(i=1 | PS)
&=& p(k_1 > k_2) \\
&=& p(w_2/u_2 \le w_1/u_1) \\
&=& p(u_1/u_2 \le w_1/w_2) \\
\text{Now, define $Z=U_1/U_2$,} \\
&=& p(Z \le w_1/w_2) \\
\end{eqnarray*}
$$

The PDF for the ratio of uniforms is
$$
p_Z(z) =
\begin{cases}
    \frac{1}{2}           & \text{if } 0 \le z \le 1 \\
    \frac{1}{2 \cdot z^2} & \text{if } z > 1         \\
    0                     & \text{otherwise}
\end{cases}
$$

We want to integrate the PDF from $z \in (0, w_1/w_2]$, which is really easy
because $0 < w_1/w_2 \le 1$ only hits the first case (a constant $1/2$).

$$
\begin{eqnarray*}
&=& \int_0^{w_1/w_2} p_Z(z) dz \\
&=& (1/2) \cdot (w_1/w_2)
\end{eqnarray*}
$$

</div>
