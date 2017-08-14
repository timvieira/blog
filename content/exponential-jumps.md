title: Exponential jumps: Reservoir sampling with much fewer random numbers
date: 2017-04-22
comments: true
status: draft
tags: sampling reservoir-sampling

The following algorithm can be used to generate a random sample from a stream of
weighted items:

```python
def weighted_reservoir(stream):
    R = None
    T = 0
    for i, w in stream:
        K = uniform(0,1) ** (1.0/w)
        if K > T:
            T = K
            R = i
    return R
```

This a pretty famous algorithm due to Eframidis and Spirakis (2006). In that
same paper they introduce a less well-known algorith, that I thought I'd
describe a bit. The idea is to reduce the number of random number's needed to
generate a sample. `weighted_reservoir` requires $\mathcal{O}(n)$ random numbers
of a stream of length $n$. The following algorithm, requires an expected number
of samples in $\mathcal{O}(\log n)$. So *way* fewer. Here is the algorith, it's
called the exponential jumps algorithm:

```python
def expjump(stream):
    stream = iter(enumerate(stream))

    R = None
    T = 0

    try:
        for i, w in stream:

            # Exponential jump. See derivation part 1.
            J = log(uniform(0,1)) / log(T) if R is not None else 0
            S = 0.0
            while S + w < J:
                S += w
                i, w = stream.next()

            # Weird key: See derivation part 2.
            T = uniform(T ** w, 1) ** (1 / w)
            R = i

    except StopIteration:
        pass

    return R

```

## Why `expjumps` works

The way we're going to understand why this algorithm works is by showing that it
simulates the first algorithm.

### Part 1: The "exponential jump"

The probability that we advance from the current position $c$ to some future
position $i$ is equal to

$$
p\left( \sum_{j=c}^{i-1} w_j < J \le \sum_{j=c}^{i} w_j \right)
$$

Let $\ell=\sum_{j=c}^{i-1} w_j$

$$
\begin{eqnarray*}
 &=& p\left(               \ell < \frac{\log(U)}{\log(T)} \le \ell + w_i         \right) \\
 &=& p\left(       \log(T) \ell < \log(U)         \le \log(T) (\ell + w_i)       \right) \\
 &=& p\left( \exp(\log(T) \ell) < U               \le \exp(\log(T) (\ell + w_i)) \right) \\
 &=& p\left(           T^{\ell} < U               \le T^{\ell + w_i} \right) \\
 &=& T^{\ell + w_i} - T^\ell
\end{eqnarray*}
$$

XXX: looks backwards, p[a < X <= b] = cdf(b) - cdf(a)

Which is equivalent to the ordinary version of the algorithm:

$$
\begin{eqnarray*}
&& \!\!\!\!\!\!\!\! p\left( \text{start at $c$ and only $i$ goes in $R$} \right) \\
&=& p\left( \text{$i$ goes in $R$} \right) \prod_{k=c}^{i-1} p\left( \text{$k$ does not go in $R$} \right) \\
&=& p\left( U_i^{1/w_i} > T \right) \prod_{k=c}^{i-1} p\left( U_k^{1/w_k} \le T \right) \\
&=& (1 - T^{w_i}) \prod_{k=c}^{i-1} T^{w_k} \\
&=& (1 - T^{w_i}) \cdot T^{\sum_{j=c}^{i-1} w_j} \\
&=& (1 - T^{w_i}) \cdot T^{\ell} \\
&=& T^{\ell} - T^{w_i} T^{\ell} \\
&=& T^\ell - T^{\ell + w_i}
\end{eqnarray*}
$$

### Part 2: Where did $T = \textrm{Uniform}(T^{w_i}, 1)^{1 / w_i}$ come from?

At a high level, the reason this expression is sort of complicated is because
$T$ is conditioned on the event $(i \in R)$.

Now, let's work out that distribution:

For notational simplify, I'll to suppress the dependence on $i$, so $K = k_i =
U_i^{1/w_i}$ and $w = w_i$.

Let's derive an inverse CDF generator conditioned on $(i \in R)$.

$$
\begin{eqnarray*}
p\left( K \le x \mid i \in R \right)
&=& p\left( K \le x \mid K > T \right) \\
&=& p\left( U^{1/w} \le x \mid U^{1/w} > T \right) \\
&=& p\left( U \le X^w \mid U > T^w \right)
\end{eqnarray*}
$$

Apply the definition of conditional probability and shift the focus to the
uniform variate U because we place in a nice cozy position between the
inequalities.

$$
   = \frac{p\left( T^w < U \le X^w \right) }{ p\left( U > T^w \right) }
$$

Solve for the numerator and denominator given $U \sim \textrm{Uniform}(0,1)$,

$$
   = \frac{X^w - T^w}{1-T^w}
$$

To apply the inverse CDF method, we solve for the target RV, $X$, in
terms of $U$.

\begin{eqnarray*}
                             U &=& \frac{X^w - T^w}{1-T^w} \\
               U \cdot (1-T^w) &=& (X^w - T^w)             \\
         U \cdot (1-T^w) + T^w &=& X^w                     \\
 (U \cdot (1-T^w) + T^w)^{1/w} &=& X
\end{eqnarray*}

In other words, $X = \textrm{Uniform}(T^w, 1)^{1/w}$.
