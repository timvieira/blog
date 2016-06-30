title: Sqrt-biased sampling
date: 2016-06-28
comments: true
status: draft
tags: sampling

The following post is about instance of "sampling in proportion to $p$ is not
optimal, but you probably think it is." It's surprising how few people seem to
know this trick. Myself included! It was brought to my attention recently by
[Nikos Karampatziakis](http://lowrank.net/nikos/). (Thanks, Nikos!)

The paper credited for this trick is
[Press (2008)](http://www.pnas.org/content/106/6/1716.full.pdf). I'm borrowing
heavily from that paper as well as an email exchange from Nikos.

**Setting**: Suppose you're trying to find a special recipe from a cookbook that
  you made one time but just can't remember exactly which recipe it was. So,
  based on the ingredients of each recipe, you come up with a prior probability
  $p_i$ that recipe $i$ is the one you're looking for. In total, the cookbook
  has $n$ recipes and $\sum_{i=1}^n p_i = 1$.

A good strategy would be to sort recipes by $p_i$ and cook the most promising
ones first. Unfortunately, you're not a great chef so there is some probability
that you make it wrong. So, it's a good idea to try recipes multiple times. This
suggests a *sampling with replacement* strategy. Let's call this strategy
$\boldsymbol{q}$ (to distinguish it from $\boldsymbol{p}$).

**How many recipes until we find the special one?** To start, suppose the
special recipe is $j$. Then, the expected number of recipes we have to make
until we find $j$ under the strategy $\boldsymbol{q}$ is

$$
\sum_{t=1}^\infty t \cdot (1 - q_j)^{t-1} q_{j} = 1/q_{j}.
$$

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
function toggle(x) { $(x).toggle(); }
</script>

<button onclick="toggle('#derivation')" class="toggle-button">Derivation</button>
<div id="derivation" style="display:none;" class="derivation">
**Derivation**:

We start with
$$
\sum_{t=1}^\infty t \cdot (1 - q_j)^{t-1} q_{j},
$$

Let $a = (1-q_j)$, to clean up notation.
$$
= q_{j} \sum_{t=1}^\infty t \cdot a^{t-1}
$$

Using the identity $\nabla_a [ a^t ] = t \cdot a^{t-1}$,
$$
= q_{j} \sum_{t=1}^\infty \nabla_a[ a^{t} ].
$$

Fishing the gradient out of the sum and tweaking summation index,
$$
= q_{j} \nabla_a\left[ \sum_{t=1}^\infty a^{t} \right]
= q_{j} \nabla_a\left[ -1 + \sum_{t=0}^\infty a^{t}\right]
$$

Plugin in the solution to the geometric series,
$$
= q_{j} \nabla_a\left[ -1 + \frac{1}{1-a} \right].
$$

Take the derivative, expanding $a$ and simplify,
$$
= q_{j} \frac{1}{(1-a)^2}
= \frac{1}{q_j}
$$

</div>


The equation says that expected time it takes to sample $j$ for *the first time*
is the probability we didn't sample for $(t-1)$ steps times the probability we
sample it at time $t$. We multiply this probability by the time $t$ to get the
*expected* time.

This equation assumes that we known $j$ is the special recipe with certainty
when we sample it. We'll revisit this assumption later when we consider
potential errors in executing the recipe.

Since we don't which $j$ is the right one, we take an expectation over it
according to the prior distribution, which yields the following equation,
$$
f(\boldsymbol{q}) = \sum_{i=1}^n \frac{p_i}{q_i}.
$$

**The first surprising thing**: Uniform is just as good as $\boldsymbol{p}$,
  yikes! $f(\boldsymbol{p}) = \sum_{i=1}^n \frac{p_i}{p_i} = n$ and
  $f(\text{uniform}(n)) = \sum_{i=1}^n \frac{p_i }{ 1/n } = n$. (Assume, without
  loss of generality, that $p_i > 0$ since we can just drop these elements from
  $\boldsymbol{p}$.)


**What's the *optimal* $\boldsymbol{q}$?** We can address this question by
solving the following optimization (which will have a nice closed form
solution),

$$
\begin{eqnarray*}
&& \boldsymbol{q}^* = \underset{\boldsymbol{q}}{\operatorname{argmin}} \sum_{i=1}^n \frac{p_i}{q_i} \\
&& \ \ \ \ \ \ \ \ \text{ s.t. } \sum_{i=1}^n q_i = 1 \\
&& \ \ \ \ \ \ \ \ \ \ \ \ \, q_1 \ldots q_n \ge 0.
\end{eqnarray*}
$$

The optimization problem says minimize the expected time to find the special
recipe. The constraints enforce that $\boldsymbol{q}$ be a valid probability
distribution.

The optimal strategy, which we get via Lagrange multipliers, turns out to be,
$$
q^*_i = \frac{ \sqrt{p_i} }{ \sum_{j=1}^n \sqrt{p_j} }.
$$


<button onclick="toggle('#Lagrange')" class="toggle-button">Derivation</button>
<div id="Lagrange" style="display:none;" class="derivation">
To solve this constrained optimization problem, we form the
Lagrangian,

$$\mathcal{L}(\boldsymbol{q}, \lambda) = \sum_{i=1}^n \frac{p_i}{q_i} - \lambda\cdot \left(1 - \sum_{i=1}^n q_i\right),$$

and solve for $\boldsymbol{q}$ and multiplier $\lambda$ such that partial
derivatives are all equal to zero. This gives us the following system of
nonlinear equations,

$$
\begin{eqnarray*}
&& \lambda - \frac{p_i}{q_i^2} = 0 \ \ \ \text{for } 1 \le i \le n \\
&& \lambda \cdot \left(1 - \sum_{i=1}^n q_i \right) = 0.
\end{eqnarray*}
$$

We see that $q_i = \pm \sqrt{\frac{p_i}{\lambda}}$ works for the first set of
equations, but since we need $q_i \ge 0$, we take the positive one. Solving for
$\lambda$ and plugging it in, we get a normalized distribution,

$$
q^*_i = \frac{ \sqrt{p_i} }{ \sum_{j=1}^n \sqrt{p_j} }.
$$

</div>


**How much better is $q^*$?**
$$
f(q^*) = \sum_i \frac{p_i}{q^*_i}
= \sum_i \frac{p_i}{ \frac{\sqrt{p_i} }{ \sum_j \sqrt{p_j}} }
= \left( \sum_i \frac{p_i}{ \sqrt{p_i} } \right) \left( \sum_j \sqrt{p_j} \right)
= \left( \sum_i \sqrt{p_i} \right)^2
$$

which sometimes equals $n$, e.g., when $\boldsymbol{p}$ is uniform, but is never
bigger than $n$.

**What's the intuition?** The reason why the $\sqrt{p}$-scheme is preferred is
because we save on *additional* cooking experiments. For example, if a recipe
has $k$ higher prior probability than the average recipe, then that recipe will
try it $\sqrt{k}$ more often; compared to instead of $k$, which we'd get under
$\boldsymbol{p}$. Additional cooking experiments are not so advantageous.

**Allowing for noise in the cooking process**: Suppose that for each recipe we
  had a prior belief about how hard that recipe is for us to cook. Denote that
  belief $s_i$ ($s$ is for "at our *skill* level"), these belief are between
  zero (never get it right) and one (perfect everytime) and do not sum to one
  over the cookbook.

Following a similar derivation to before, the time to cook the special recipe
$j$ and cook it correctly is,
$$
\sum_{t=1}^\infty t \cdot (1 - \color{red}{s_j} q_j)^{t-1} q_{j} \color{red}{s_j} = \frac{1}{s_j \cdot q_j}
$$
That gives rise to a modified objective,
$$
f'(\boldsymbol{q}) = \sum_{i=1}^n \frac{p_i}{\color{red}{s_i} \cdot q_i}
$$

This is exactly the same as the previous objective, except we've replaced $p_i$
with $p_i/s_i$. Thus, we can reuse our previous derivation to get the optimal
strategy, $q^*_i \propto \sqrt{p_i / s_i}$. If noise is constant, then we
recover the original solution, $q^*_i \propto \sqrt{p_i}$.

**A more realistic application**: In certain language modeling applications, we
  avoid computing normalization constants by using importance sampling or noise
  contrastive estimation techniques. These techniques depend on a proposal
  distribution, which folks often take to be the unigram
  distribution. Unfortunately, this gives too many samples of stop words (e.g.,
  "the", "an", "a"), so practitioners "anneal" the unigram distribution (to
  increase the entropy), that is sample from $q_i \propto
  p_{\text{unigram},i}^\alpha$. Typically, $\alpha$ is set by grid search and,
  no surprise, $\alpha \approx 1/2$ tends to work best! The $\sqrt{p}$-sampling
  trick is a theoretical justification in favor of annealing by $1/2$ as well as
  annealing as the right thing to at all (e.g., why not do additive
  smoothing?). The assumption is that we want to sample the actual word at a
  given position (the special recipe) as soon as possible, given that all we
  have access to is the unigram prior.
