title: Faster reservoir sampling by waiting
date: 2019-06-11
comments: true
tags: sampling, reservoir-sampling, Gumbel, sampling-without-replacement

We are interested in designing an efficient algorithm for sampling from a categorical distribution over $n$ items with weights $w_i > 0$.  Define target sampling distribution $p$ as
$$
p = \mathrm{Categorical}\left( \frac{1}{W} \cdot \vec{w} \right)
\quad\text{where}\quad W = \frac{1}{\sum_j w_j}
$$

The following is a very simple and relatively famous algorithm due to [Efraimidis and Spirakis (2006)](https://www.sciencedirect.com/science/article/pii/S002001900500298X).  It has several useful properties (e.g., it is a one-pass "streaming" algorithm, separates data from noise, can be easily extended for streaming sampling without replacement).  It is also very closely related to the Gumbel-max trick (discussed in [Vieira, 2014](http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/)).


```python
def weighted_reservoir_sampling(stream):
    return np.argmin([Exponential.sample(w) for w in stream])
```

Some differences from E&S'06: Our $K$ and $T$ are $\log$ of the $K$ and $T$ variables in E&S'06.  Additionally, we take $\min$ instead of $\max$.  These differences allow us to talk about exponential variates rather than the less-elegant and rather-mysterious (IMO) random key $u_i^{1/w_i}$.

**Why does it work?** The weighted-reservoir sampling algorithm exploits the following well-known properties of exponential random variates:
When $X_i \sim \mathrm{Exponential}(w_i)$, $R = {\mathrm{argmin}}_i X_i$, and $T = \min_i X_i$ then
$R \sim p$ and $T \sim \mathrm{Exponential}\left( \sum_i w_i \right)$.


## Fewer random variates by waiting

One down-side of this one-pass algorithm is that it requires $\mathcal{O}(n)$ uniform random variates.  Contrast that with the usual, two-pass methods for sampling from a categorical distribution, which only need $\mathcal{O}(1)$ samples.  E&S'06 also present a much less well-known algorithm, called the "Exponential jumps" algorithm, which is a one-pass algorithm that only requires $\mathcal{O}(\log(n))$ random variates (in expectation).  That's *way* fewer random variates and a small price to pay if you are trying to avoid paging-in data from disk a second time.

Here is my take on their algorithm.  There is no substantive difference, but I believe my version is more instructive since it makes the connection to exponential variates and truncated generation explicit (i.e., no mysterious random keys).

```python
def jump(stream):
    "Weighted-reservoir sampling by jumping"
    R = None
    T = np.inf
    J = 0.0
    for i, w in enumerate(stream):
        J -= w
        if J <= 0:
            # Sample the key for item i, given that it is smaller than the current threshold
            T = Exponential.sample_truncated(w, 0, T)
            # i enters the reservoir
            R = i
            # sample the waiting time (size of the jump)
            J = Exponential.sample(T)
    return R
```

**Why does exponential jumps work?**

Let me first write the `weighted_reservoir_sampling` algorithm to be much more similar to the `jump` algorithm.  For fun, I'm going to refer to it as the `walk` algorithm.

```python
def walk(stream):
    "Weighted-reservoir sampling by walking"
    R = None
    T = np.inf
    J = 0.0
    for i, w in enumerate(stream):
        X = Exponential.sample(w)
        if X < T:
            R = i   # i enters the reservoir
            T = X   # threshold to enter the reservoir
    return R
```

**The key idea** of the exponential jumps algorithm is to sample *waiting times* between new minimum events.  In particular, if the algorithm is at step $i$ the probability that sees its next minimum at steps $j \in \{ i+1, \ldots \}$ can be reasoned about without needing to *actually* sample the various $X_j$ variables.

Rather than going into a full-blown tutorial on waiting times of exponential variates, I will get to the point and show that the `jump` algorithm simulates the `walk` algorithm.  The key to doing this is showing that the probability of jumping from $i$ to $k$ is the same as "walking" from $i$ to $k$.  Let $W_{i,k} = \sum_{j=i}^k w_j$.

This proof is adapted from the original proof in E&S'06.

$$
\begin{eqnarray}
\mathrm{Pr}\left( \text{walk to } k \mid i,T \right)
&=& \mathrm{Pr}\left( X_k < T \right) \prod_{j=i}^{k-1} \mathrm{Pr}\left( X_j \ge T \right) \\
&=& \left(1 - \exp\left( T w_k \right) \right) \prod_{j=i}^{k-1} \exp\left( T w_j \right) \\
&=& \left(1 - \exp\left( T w_k \right) \right) \exp\left( T \sum_{j=i}^{k-1}  w_j \right) \\
&=& \left(1 - \exp\left( T w_k \right) \right) \exp\left( T W_{i,k-1} \right) \\
&=& \exp\left( T W_{i,k-1} \right) - \exp\left( T w_k \right) \exp\left( T W_{i,k-1} \right) \\
&=& \exp\left( T W_{i,k-1} \right) - \exp\left( T W_{i,k} \right) \\
\\
\mathrm{Pr}\left( \text{jump to } k \mid i, T \right)
&=& \mathrm{Pr}\left( W_{i,k-1} < J \le W_{i,k} \right) \\
&=& \mathrm{Pr}\left( W_{i,k-1} < -\frac{\log(U)}{T} \le W_{i,k} \right) \\
&=& \mathrm{Pr}\left( \exp(-T \cdot W_{i,k-1}) > U \ge \exp(-T \cdot W_{i,k}) \right) \label{foo}\\
&=& \exp(T \cdot W_{i,k-1}) - \exp(T \cdot W_{i,k} )
\end{eqnarray}
$$

Given that the waiting time correctly matches the walking algorithm, the remaining detail is to check that $X_k$ is equivalent under the condition that it goes into the reservoir.  This conditioning is why the jumping algorithm must generate a *truncated* random variate: a random variate that is guaranteed to less than the previous minimum.  In the [Gumbel-max world](https://cmaddis.github.io/gumbel-machinery), this is used in the top-down generative story.


## Closing thoughts

Pros:

- The jump algorithm saves a ton of random variates and gives practical savings
  (at least, in my limited experiments).

Cons:

- The jump algorithm is harder to parallelize or vectorize, but it seems possible.

- If you aren't in a setting that requires a one-pass algorithm or some other
  special properties, you are probably better served by the two-pass algorithms
  since they have lower overhead because it doesn't call expensive functions
  like $\log$ and it uses a single random variate per sample.

Further reading:

- I have several posts on the topic of fast sampling algorithms
([1](http://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/),
[2](http://timvieira.github.io/blog/post/2016/07/04/fast-sigmoid-sampling/),
[3](http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/)).

- Jake VanderPlas (2018) [The Waiting Time Paradox, or, Why Is My Bus Always Late?](http://jakevdp.github.io/blog/2018/09/13/waiting-time-paradox/).


## Interactive Notebook

<script src="https://gist.github.com/timvieira/44edfaf97cb2e191e4618f0d25401bf4.js"></script>
