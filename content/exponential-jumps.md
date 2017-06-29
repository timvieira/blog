title: Exponential jumps algorithm: reservoir sampling with fewer random numbers
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
simulate the first algorithm.

### Part 1: The "exponential jump"

The probability that we advance from c to i, is equal to

$$
  p\left( \sum_{j=c}^{i-1} w_j < J \le \sum_{j=c}^{i} w_j \right)
$$
Let $L=\sum_{j=c}^{i-1} w_j$

$$
\begin{eqnarray}
 &=& p\left(               L < \log(U)/\log(T) \le L + w[i]                 \right) \\
 &=& p\left(       \log(T) L < \log(U)         \le \log(T) (L + w[i])       \right) \\
 &=& p\left( \exp(\log(T) L) < U               \le \exp(\log(T) (L + w[i])) \right) \\
 &=& p\left(           T^{L} < U               \le T^{L + w_i} \right) \\
 &=& T^{L + w_i} - T^L
\end{eqnarray}
$$

XXX: looks backwards, p[a < X <= b] = cdf(b) - cdf(a)

Which is equivalent to the maximization version

$$
\begin{eqnarray}
&& \!\!\!\!\!\!\!\! p\left( \text{start at $c$ and only $i$ goes in $R$} \right) \\
&=& p[ \text{$i$ goes in $R$} ] \prod_{k=c}^{i-1} p[ \text{$k$ does not go in $R$} ] \\
&=& p[ U_i^{1/w_i} > T ] \prod_{k=c}^{i-1} p[ U_k^{1/w_k} \le T ] \\
&=& (1 - T^{w_i}) \prod_{k=c}^{i-1} T^{w_k} \\
&=& (1 - T^{w_i}) \cdot T^{\sum_{j=c}^{i-1} w_j} \\
&=& (1 - T^{w_i}) \cdot T^{L} \\
&=& T^{L} - T^{w_i} T^{L} \\
&=& T^L - T^{L + w_i}
\end{eqnarray}
$$

### Part 2: The weird key

Why do we have the weirdo key

  T = uniform(T ** w[i], 1) ** (1.0 / w[i]),

when we ordinarily have the following key for item i.

  k[i] = uniform(0, 1) ** (1.0 / w[i])

The intuition is that we need to generate the key *conditioned on the fact that
`i` is indeed entering R*.

I'll work this out below.

(For notational simplify, I'll to suppress the dependence on i, so k = k[i] and
w = w[i].)

Let's derive an inverse CDF generator conditioned on `i in R`.

  P[ k <= X | i in R ]
   = P[ k <= X | k > T ]
   = P[ U^1/w <= X | U^1/w > T ]
   = P[ U <= X^w | U > T^w ]

Apply the definition of conditional probability and shift the focus to the
uniform variate U because we place in a nice cozy position between the
inequalities.

   = P[ T^w < U <= X^w ] / P[U > T^w]

Solve for the numerator and denominator given U ~ Uniform(0,1),

   = (X^w - T^w) / (1-T^w)

To apply the inverse CDF method, we solve for the target RV, X, in
terms of U.

                         U = (X^w - T^w) / (1-T^w)
               U * (1-T^w) = (X^w - T^w)
         U * (1-T^w) + T^w = X^w
 (U * (1-T^w) + T^w)^(1/w) = X

Let a = T^w,

  X = (U * (1-a) + a)^(1/w) = U(a,1)^(1/w)

which is exactly what we were looking for!

  X = U(T^w, 1)^(1/w)    ∎
