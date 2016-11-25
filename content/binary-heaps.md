title: Binary heaps, incremental maintenance
date: 2016-11-21
comments: true
tags: math, sampling, datastructures


In this post, I'll describe a neat trick for maintaining a summary quantity,
such as the sum, max, product of vector of inputs. The trick and it's
implementation are very similar to the max-heap that many people are familiar
with.

I'll describe one really neat application to fast sampling under an evolving
discrete distribution using a binary heap and binary search.


**Setup**: Suppose we'd like to efficiently compute a summary quantity under
changes it's inputs $\boldsymbol{w} \in \boldsymbol{K}^n$. The particular form
of the quantity we're going to compute is

$$
z = \bigoplus_{i=1}^n w_i
$$

Where $\oplus$ is some associative binary operator $\oplus: \boldsymbol{K}
\times \boldsymbol{K} \mapsto \boldsymbol{K}$.

 * Associative: $(a \oplus b) \oplus c = a \oplus (b \oplus c)$ for all $a,b,c
   \in \boldsymbol{K}$.

From simplicity, I'm going to assume

 * An identity element $\boldsymbol{0} \in \boldsymbol{K}$ such that $k \oplus \boldsymbol{0}
   = \boldsymbol{0} \oplus k = k$, for all $k \in \boldsymbol{K}$.

 * The dimensionality of $|\boldsymbol{w}|=n$ is fixed and that it's a simple
   dense vector. Generalizations are discussed later.

**Examples** (operator / identity): $+/0$, $\times/1$, $\max/\!-\!\!\infty$,
  $\text{logadd}/\!-\!\!\infty$.

<!--
In many cases, $\oplus$ has an inverse $\ominus$ where $c = (a \oplus b)$, $c
\in \boldsymbol{K}$, $b = a \ominus (a \oplus b)$ and $c \ominus b = a$.

Thinking beyond sum. Not all operators have an inverse!

Let
$$
z = \sum_{i=1}^n w_i
$$

Suppose we change $w_k$,

Version 1:
$$
z^{\text{new}} = z^{\text{old}} - w_k^{\text{old}} + w_k^{\text{new}}
$$

A famous example is max, $z = \max_{i=1}^n w_i$. With max is no (general)
inverse operator.

  Most people learning the max-heap data structure in CS101 for doing exactly
  this. Although, it is only taught for max, so maybe you haven't thought about
  it more abstractly.

Another example, logadd. Using it's inverse, logsub, is very unstable. Take
$\boldsymbol{w} = \gamma \cdot \boldsymbol{w}'$. As $\gamma \rightarrow \infty,$
$z \rightarrow \max_i w'_i$. Very few 'bits of information' from elements other
than the max are available in z.

-->

Essentially, the trick boils down to parenthesis placement in an aggregation of
$n$ elements.

We generally compute things as a right-branching binary tree because it's easy
to write as a loop or with fold. I'll call the "linear" structure.

Here's an example with $n=8$.

$$
z = (((((((w_1 \oplus w_2) \oplus w_3) \oplus w_4) \oplus w_5) \oplus w_6) \oplus w_7) \oplus w_8)
$$

but, it is often beneficial to compute things as a balanced binary tree, which
I'll refer to as a "heap" from here on,

$$
z = (((w_1 \oplus w_2) \oplus (w_3 \oplus w_4)) \oplus ((w_5 \oplus w_6) \oplus (w_7 \oplus w_8)))
$$

The benefit of the heap structure is that there are $\mathcal{O}(\log n)$
intermediate quantities that depend on an input, whereas the linear structure
has $\mathcal{O}(n)$.

Since fewer intermediate quantities depend on a given input, fewer intermediates
need to be adjusted upon a change to the input. Therefore, we get faster
algorithms for maintaining the desired quantity as the inputs change.


<!--
 * The datastructure: Bottom-most level of the heap has size n' =
   next-power-of-two(n). The number of internal nodes in the tree is n'-1. We
   have a dummy node at position zero to make indexing math simpler. So, we
   allocate twice the size of the bottom level to fit internal nodes. Thus, the
   overal data structure is <4*n in the worst case because the next power of two
   <2n and then we have another factor of two for internal nodes.
-->

The ideas, we're keep around intermediate quantities, corresponding to each of
the parentheses in the heap. We're going to pack everything into an array (this
array is no more than $4 n$)

This heap is a *complete* binary tree of depth $\ceil{\log_2 n}$, with the
values of $\boldsymbol{w}$ at it's leaves and padding of $\boldsymbol{0}$ for
remaining leaves. We're using this structure because it makes our implementation
really nice.

Since the structure of the heap we don't need pointer or anything like that to
find the parent or children of a node.

Let $d$ be the number of internal nodes, nodes $i \le d$ are interal. Node $0$
is a dummy node, which we add to make the indexing math simpler.

For node $i$

 * left child: ${2 \cdot i}$ for $i \le d$

 * right child: ${2 \cdot i + 1}$ for $i \le d$

 * parent: $\floor{i / 2}$ if $i > 1$


**Remarks**:

 * In the case of a max or min heap, we can avoid allocating extra space for
   intermediate quantities because all intermediates values are equal to exactly
   one element of $\boldsymbol{w}$.

 * If the operations are noisy (e.g., floating point operator), then the heap
   version may be better behaved. For example, if operations have an
   independent, additive noise rate $\varepsilon$ then noise of
   $z_{\text{heap}}$ is $\mathcal{O}(\varepsilon \cdot \log n)$, whereas
   $z_{\text{linear}}$ is $\mathcal{O}(\varepsilon \cdot n)$.

<!--
 * Note that changeprop need to have a heap structure, even the right-branching
   structure is amenable to changeprop, but the heap-structured code is likely
   to be faster because we can bound the amount stuff we'll need to adjust.
-->

**Generalizations**:

 * No zero? No problem. Augment $\boldsymbol{K} \cup \{ \textsf{null} \}$ where
   $\textsf{null}$ is distinguished value that *acts* just like a zero. All
   algorithms will continue to work. We don't *actually* require a zero element.

 * Generalization to an arbitrary maps instead of fixed vectors is possible with
   a bijective "locator" map which tracks the location of elements. Any problem
   in computer science can be solved with another level of indirection.

 * Allowing $w$ to shink

    - We can replace the old value with $\boldsymbol{0}$ or $\textsf{null}$. If
      the $\boldsymbol{w}$ ever hits $>50\%$ $\textsf{null}$, we can perform an
      explicit shrink operations (at no asymptotic cost).

    - If $\oplus$ is commutative, you can swap the last element of the array into
      the deleted position and replace the last element with $\boldsymbol{0}$ or
      $\textsf{null}$.

 * Allowing $w$ to grow: similar to shrinking, growing the vector can be done by
   doubling the underlying array and filling the unused portion with
   $\textsf{null}$ or $\boldsymbol{0}$.


**Use cases**

 * Gibbs sampling?

 * EXP3 (bandit algorithm) is an excellent example of an algorithm that samples
   and modifies a distribution as it runs.

 * Stochastic priority queues where we sample proportional to priority and the
   weights on items in the queue change, elements are possibly removed after
   they are sampled (i.e., sampling without replacement), elements are added.


```python
import numpy as np
from numpy.random import uniform


def update(S, k, v):
    "Update heap `S` at position `k` with value `v` in time O(log n)."
    d = S.shape[0]
    # change
    i = d//2 + k
    S[i] = v
    # propagate
    while i > 0:
        i //= 2
        S[i] = S[2*i] + S[2*i + 1]


def sumheap(w):
    "Create sumheap from weights `w`."
    n = w.shape[0]

    d = int(2**np.ceil(np.log2(n)))
    S = np.zeros(2*d)

    # O(n) version (faster than calling update n times => O(n log n))
    S[d:d+n] = w
    for i in reversed(range(1, d)):
        S[i] = S[2*i] + S[2*i + 1]

    return S


def check(S, i):
    "Check heap invariant."
    d = S.shape[0]
    if i >= d//2:   # only checks internal nodes.
        return
    assert S[i] == S[2*i] + S[2*i + 1]
    check(S, 2*i)
    check(S, 2*i + 1)


def dump(S):
    "Print heap for debugging."
    for i in range(int(np.ceil(np.log2(len(S))))):
        print 'depth', i, S[2**i:2**(i+1)]


def sample(w, u):
    "Ordinary sampling method, O(n) to build heap, O(log n) per sample after that."
    c = w.cumsum()
    return c.searchsorted(u * c[-1])


def hsample(S, u):
    "Sample from sumheap, O(log n) per sample."
    offset = S.shape[0]//2  # number of internal nodes.
    # random probe
    p = S[1] * u
    # Use binary search to find the index of the largest CDF (represented as a
    # heap) value that is less than a random probe.
    i = 1
    while i < offset:
        # Determine if the value is in the left or right subtree.
        i *= 2
        left = S[i]
        if p > left:
            # Value is in right subtree. Subtract mass under left subtree.
            p -= left
            i += 1
    return i - offset


def main():
    for n in np.random.choice(range(1, 100), size=10):
        print n
        w = np.round(uniform(0, 10, size=n), 1)
        S = sumheap(w)
        check(S, 1)
        for _ in range(100):
            u = uniform()
            p1 = sample(w, u)
            p2 = hsample(S, u)
            assert p1 == p2
            # change a random value in the weight array
            c = np.random.randint(n)
            v = uniform(10)
            w[c] = v
            update(S, c, v)
            check(S, 1)


if __name__ == '__main__':
    main()
```
