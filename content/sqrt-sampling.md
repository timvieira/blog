title: Sqrt-biased sampling
date: 2016-06-28
comments: true
tags: sampling, decision-making

The following post is about instance of "sampling in proportion to $p$ is not
optimal, but you probably think it is." It's surprising how few people seem to
know this trick. Myself included! It was brought to my attention recently by
[Nikos Karampatziakis](http://lowrank.net/nikos/). (Thanks, Nikos!)

The paper credited for this trick is
[Press (2008)](http://www.pnas.org/content/106/6/1716.full.pdf). I'm borrowing
heavily from that paper as well as an email exchange from Nikos.

**Setting**: Suppose you're an aspiring chef with a severe head injury affecting
  your long- and short- term memory trying to find a special recipe from a
  cookbook that you made one time but just can't remember exactly which recipe
  it was. So, based on the ingredients of each recipe, you come up with a prior
  probability $p_i$ that recipe $i$ is the one you're looking for. In total, the
  cookbook has $n$ recipes and $\sum_{i=1}^n p_i = 1.$

A good strategy would be to sort recipes by $p_i$ and cook the most promising
ones first. Unfortunately, you're not a great chef so there is some probability
that you'll mess-up the recipe. So, it's a good idea to try recipes multiple
times. Also, you have no short term memory...

This suggests a *sampling with replacement* strategy, where we sample a recipe
from the cookbook to try *independently* of whether we've tried it before
(called a *memoryless* strategy). Let's give this strategy the name
$\boldsymbol{q}.$ Note that $\boldsymbol{q}$ is a probability distribution over
the recipes in the cookbook, just like $\boldsymbol{p}.$

**How many recipes until we find the special one?** To start, suppose the
special recipe is $j.$ Then, the expected number of recipes we have to make
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
// workaround for when markdown/mathjax gets confused by the
// javascript dollar function.
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

Use the identity $\nabla_a [ a^t ] = t \cdot a^{t-1}$,
$$
= q_{j} \sum_{t=1}^\infty \nabla_a[ a^{t} ].
$$

Fish the gradient out of the sum and tweak summation index,
$$
= q_{j} \nabla_a\left[ \sum_{t=1}^\infty a^{t} \right]
= q_{j} \nabla_a\left[ -1 + \sum_{t=0}^\infty a^{t}\right]
$$

Plugin in the solution to the geometric series,
$$
= q_{j} \nabla_a\left[ -1 + \frac{1}{1-a} \right].
$$

Take derivative, expand $a$ and simplify,
$$
= q_{j} \frac{1}{(1-a)^2}
= \frac{1}{q_j}
$$

</div>


The equation says that expected time it takes to sample $j$ for *the first time*
is the probability we didn't sample for $(t-1)$ steps times the probability we
sample it at time $t.$ We multiply this probability by the time $t$ to get the
*expected* time.

Note that this equation assumes that we known $j$ is the special recipe *with
certainty* when we sample it. We'll revisit this assumption later when we
consider potential errors in executing the recipe.

Since we don't known which $j$ is the right one, we take an expectation over it
according to the prior distribution, which yields the following equation,
$$
f(\boldsymbol{q}) = \sum_{i=1}^n \frac{p_i}{q_i}.
$$

**The first surprising thing**: Uniform is just as good as $\boldsymbol{p}$,
  yikes! $f(\boldsymbol{p}) = \sum_{i=1}^n \frac{p_i}{p_i} = n$ and
  $f(\text{uniform}(n)) = \sum_{i=1}^n \frac{p_i }{ 1/n } = n.$ (Assume, without
  loss of generality, that $p_i > 0$ since we can just drop these elements from
  $\boldsymbol{p}.$)


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
bigger than $n.$

**What's the intuition?** The reason why the $\sqrt{p}$-scheme is preferred is
because we save on *additional* cooking experiments. For example, if a recipe
has $k$ times higher prior probability than the average recipe, then we will try
that recipe $\sqrt{k}$ times more often; compared to $k$, which we'd get under
$\boldsymbol{p}.$ Additional cooking experiments are not so advantageous.

**Allowing for noise in the cooking process**: Suppose that for each recipe we
  had a prior belief about how hard that recipe is for us to cook. Denote that
  belief $s_i$, these belief are between zero (never get it right) and one
  (perfect every time) and do not sum to one over the cookbook.

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
with $p_i/s_i.$ Thus, we can reuse our previous derivation to get the optimal
strategy, $q^*_i \propto \sqrt{p_i / s_i}.$ If noise is constant, then we
recover the original solution, $q^*_i \propto \sqrt{p_i}.$


**Extension to finding multiple tasty recipes**: Suppose we're trying to find
  several tasty recipes, not just a single special one. Now, $p_i$ is our prior
  belief that we'll like the recipe at all. How do we minimize the time until we
  find a tasty one? It turns out the same trick works without modification
  because all derivations apply to each recipe independently. The same trick
  works if $p_i$ does not sums to one over $n.$ For example, if $p_i$ is the
  independent probability that you'll like recipe $i$ at all, not the
  probability that it's the special one.

**Beyond memoryless policies**: Clearly, our choice of a memoryless policy can
  be beat by a policy family that balances exploration (trying new recipes) and
  exploitation (trying our best guess).

  * Overall, the problem we've posed is similar to a
    [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit). In
    our case, the arms are the recipes, pulling the arm is trying the recipe and
    the reward is whether or not we liked the recipe (possibly noisy). The key
    difference between our setup and multi-armed bandits is that we trust our
    prior distribution $\boldsymbol{p}$ and noise model $\boldsymbol{s}.$

  * If the amount of noise $s_i$ is known and we trust the prior $p_i$ then
    there is an optimal deterministic (without-replacement) strategy that we can
    get by sorting the recipes by $p_i$ accounting for the error rates
    $s_i.$ This approach is described in the original paper.

**A more realistic application**: In certain language modeling applications, we
  avoid computing normalization constants (which require summing over a massive
  vocabulary) by using importance sampling, negative sampling or noise
  contrastive estimation techniques (e.g.,
  [Ji+,16](https://arxiv.org/pdf/1511.06909.pdf)
  [Levy+,15](http://www.aclweb.org/anthology/Q15-1016)). These techniques depend
  on a proposal distribution, which folks often take to be the unigram
  distribution. Unfortunately, this gives too many samples of stop words (e.g.,
  "the", "an", "a"), so practitioners "anneal" the unigram distribution (to
  increase the entropy), that is sample from $q_i \propto
  p_{\text{unigram},i}^\alpha.$ Typically, $\alpha$ is set by grid search and
  (no surprise) $\alpha \approx 1/2$ tends to work best! The $\sqrt{p}$-sampling
  trick is possibly a reverse-engineered justification in favor of annealing as
  "the right thing to do" (e.g., why not do additive smoothing?) and it even
  tells us how to set the annealing parameter $\alpha.$ The key assumption is
  that we want to sample the actual word at a given position as often as
  possible while still being diverse thanks to the coverage of unigram
  prior. (Furthermore, memoryless sampling leads to simpler algorithms.)

<!--
Actually, many word2vec papers use $\alpha=3/4$, which was suggested in
[Levy+,15](http://www.aclweb.org/anthology/Q15-1016), including the default
value in
[gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L462). So,
[Ryan Cotterell](https://ryancotterell.github.io/) ran a quick experiment with
gensim, which confirmed the suspicion that $1/2$ may be better than $3/4.$

    Word similarity accuracy (avg of 10 runs)
    | alpha | accuracy |
    +==================+
    |  0.00 |    0.354 |
    |  0.25 |    0.403 |
    |  0.50 |    0.414 |
    |  0.75 |    0.395 |
    |  1.00 |    0.345 |
-->
