title: Learning from bandit feedback
date: 2017-11-30
comments: true
tags: counterfactual-reasoning, importance-sampling, machine-learning, bandit, rl
status: draft

<style> .toggle-button { background-color: #555555; border: none; color: white;
padding: 10px 15px; border-radius: 6px; text-align: center; text-decoration:
none; display: inline-block; font-size: 16px; cursor: pointer; } .derivation {
background-color: #f2f2f2; border: thin solid #ddd; padding: 10px;
margin-bottom: 10px; } </style>
<script>
/* workaround for when markdown/mathjax gets confused by the javascript dollar function. */
function toggle(x) { $(x).toggle(); }
</script>


At the heart of reinforcement learning lies the Robert Frost dilemma where an
agent finds themselves remis about a path not taken (<code>#yolo</code>). In
machine learning, this is know as bandit feedback, or learning from a partial
response (and it probably has many other names).

I have written about
[counterfactual reasoning](post/2016/12/19/counterfactual-reasoning-and-learning-from-logged-data/)
before.

In this post, I'll dive into partial feedback in more depth.

It seems like people are a bit confused about "action-dependent baselines"


Action-dependent baselines are a simple application of the control-variate
technique.

$$
\hat{r}(s_t, A) = \Big( r(s_t, a_t) - c(s_t, a_t) \Big) \frac{I(A = a_t) }{ p(a_t|s_t) } + c(s_t, A)
$$

Let's get rid of the clutter to do with $t$ and $s_t$, since it does not matter
for our discussion.

$$
\hat{r}(A) = \Big( r(a) - c(a) \Big) \frac{I(A = a) }{ p(a) } + c(A)
$$

$\hat{r}$ is a function of actions we could have taken, but the action $a$ that
we actually took is fixed.


It's clear that this estimate is unbiased:

$$
\begin{eqnarray*}
\mathbb{E}\left[ \hat{r}(s_t, A) \right]
&=& \mathbb{E}\left[ \Big( r(s_t, a_t) - c(s_t, a_t) \Big) \frac{I(A = a_t) }{ p(a_t \mid s_t) } + c(s_t, A) \right] \\
&=& \mathbb{E}\left[ r(s_t, a_t)\frac{I(A = a_t) }{ p(a_t \mid s_t) } \right] - \mathbb{E}\left[ c(s_t, a_t) \frac{I(A = a_t) }{ p(a_t \mid s_t) } \right] + \mathbb{E}\left[ c(s_t, A) \right] \\
&=& \mathbb{E}\left[ r(s_t, A) \right] - \mathbb{E}\left[ c(s_t, a_t) \right] + \mathbb{E}\left[ c(s_t, A) \right] \\
&=& \mathbb{E}\left[ r(s_t, A) \right]
\end{eqnarray*}
$$

Note that when we go any use $\hat{r}$, we will marginalize over actions.

To estimate the gradient, $\mathbb{E}[r(A)] \approx \sum_{A} p(A | s_t)
\hat{r}(s_t, A) \nabla_{\theta} \log p(A | s_t)$, which is unbiased thanks to
the linearity of expectation.

To extend this to RL, simple take $r$ to be any unbiased estimate of the
action-value function, $\hat{q}_\pi(s_t, a_t)$ and $c(s, A)$ to be any correlate
thereof, including even a scalar in order to reduce variance.

The way to think about policy gradient is that it computes the gradient of the
"local risk". Where the reward is estimated by $\hat{r}(s_t, A)$. As long as the
reward estimate is unbiased, the policy gradient estimate will be unbiased.

Actually, the gradient of risk is invariant to a *scalar* shift in $\hat{r}$,
$grad(\theta, (\hat{r}(\cdot) + \Delta) = grad(\theta, \hat{r}(\cdot))$ for any
$\Delta$.

This is because of linearity of expectation.

this is because the policy is locally normalize to a give state, $\sum_a \pi( a
\mid s) = 1$ (or with an integral if $a$ is continuous).

Of course, it might be inefficient to marginalize over all actions, but
importantly there is not added sample complexity to do so (all quantities
involved are known), the only cost is computational.

Of course, it might in silly to marginalize over all actions when $\hat{r}$ is a
one-hot vector.
