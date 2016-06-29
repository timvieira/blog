title: Sqrt-biased sampling
date: 2016-06-28
comments: true
status: draft
tags: sampling

The following post is about instance of "sampling from $p$ is not optimal, but
you probably think it is." It's surprising how few people seem to know this
trick. Myself included! It was brought to my attention recently by
[Nikos Karampatziakis](http://lowrank.net/nikos/). (Thanks, Nikos!)

The paper credited for this trick is
[Press (2008)](http://www.pnas.org/content/106/6/1716.full.pdf). I'm borrowing
heavily from that paper as well as an email exchange from Nikos.

**Setting**: Suppose you are a government trying to identify terrorists. You
have hired a team of data scientists to develop a fancy machine learning model,
which assigns a probability $p_i$ that individual $i$ of the population (of size
$n$) is a terrorist.

Since lining people up and testing them is not very nice, we decide to sample
individuals (with replacement) for a random screening.

**How many individuals do we test to find one terrorist?** Suppose individual
$j$ is the terrorist. The expected number of screenings until we find $j$ under
a randomized screening policy $q$ (possibly different from $p$) is
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

which "marginalizes out" the time $t$ we find the terrorist. Each term
multiplies the time step by the probability we find the terrorist at exactly
that time.

Let $a = (1-q_j)$, to clean up notation slightly.
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

If the terrorist $j$ is actually distributed according to $p$, then the
expected number of screenings per terrorist found is
$$
f(\boldsymbol{q}) = \sum_{i=1}^n \frac{p_i}{q_i}.
$$

Let's plugin a few values in for $\boldsymbol{q}$.

$f(\boldsymbol{p}) = \sum_{i=1}^n \frac{p_i}{p_i} = n$.

$f(\text{Uniform}(n)) = \sum_{i=1}^n \frac{p_i }{ 1/n } = n$

Uniform is just as good as $\boldsymbol{p}$, yikes!

**What's the *optimal* $\boldsymbol{q}$?**: We can address this question by
solving the following optimization (which will have a nice closed form
solution),

$$
\begin{eqnarray*}
&& \underset{\boldsymbol{q}}{\operatorname{argmin}} \sum_{i=1}^n \frac{p_i}{q_i} \\
&& \ \ \ \ \ \ \ \ \text{ s.t. } \sum_{i=1}^n q_i = 1 \\
&& \ \ \ \ \ \ \ \ \ \ \ \ \, q_1 \ldots q_n \ge 0.
\end{eqnarray*}
$$

The objective minimizes the number of screenings per terrorist found subject to
the constraint that $\boldsymbol{q}$ is a valid probability distribution. The
solutions turns out to be

$$
q_i = \frac{ \sqrt{p_i} }{ \sum_{j=1}^n \sqrt{p_j} }.
$$


<button onclick="toggle('#Lagrange')" class="toggle-button">Derivation</button>
<div id="Lagrange" style="display:none;" class="derivation">
To solve this constrained optimization problem, we form the
Lagrangian,

$$\mathcal{L}(\boldsymbol{q}, \lambda) = \sum_{i=1}^n \frac{p_i}{q_i} - \lambda\cdot (1 - \sum_{i=1}^n q_i),$$

and solve for $\boldsymbol{q}$ and Langrange multiper $\lambda$ such that
partial derivatives are all equal to zero, which gives us the following system
of nonlinear equations:

$$
\begin{eqnarray*}
&& \lambda - \frac{p_i}{q_i^2} = 0 \ \ \ \text{for } 1 \le i \le n \\
&& \lambda \cdot (1 - \sum_{i=1}^n q_i) = 0
\end{eqnarray*}
$$

We see that $q_i = \pm \sqrt{\frac{p_i}{\lambda}}$ work for the first equation,
but since we need $q_i \ge 0$, we take the positive one. Plugging in $\lambda$,
we get a normalized distribution,

$$
q_i = \frac{ \sqrt{p_i} }{ \sum_{j=1}^n \sqrt{p_j} }.
$$

</div>


**How much better is $f(q^*)$ vs $f(p)$?**
$$
f(q^*) = \sum_i \frac{p_i}{q^*_i}
= \sum_i \frac{p_i}{ \frac{\sqrt{p_i} }{ \sum_j \sqrt{p_j}} }
= \left( \sum_i \frac{p_i}{ \sqrt{p_i} } \right) \left( \sum_j \sqrt{p_j} \right)
= \left( \sum_i \sqrt{p_i} \right)^2
$$

which sometimes equals $n$, e.g., when $\boldsymbol{p}$ is uniform, but is never
bigger than $n$.

**What's the intuition?**: The reason why the $\sqrt{p}$-scheme is preferred is
because we save on *additional* screenings. For example, if an individual has
$k$ times higher risk than the average individual, then that person will screen
$\sqrt{k}$ more often (instead of $k$, which we'd get under $\boldsymbol{p}$).

We can improve upon the $\sqrt{p}$-scheme by sampling *without replacement*, but
that many be undesirable for many reasons. The ideal without replacement scheme
would be to sort the individuals according to $\boldsymbol{p}$ (in increasing
order), which would give us $\sum_{i=1}^n i \cdot p_{\text{sorted}(i)}$. This
can be significantly better than the $\sqrt{p}$-scheme. Some theoretical
comparison is available in the original paper.

**Summary**: Even the math says that you shouldn't be profiling that hard! Take
  your prejudices and $\sqrt{\cdot}$ them for a more productive life! Remember
  that $p$ is tempting, but often not optimal!
