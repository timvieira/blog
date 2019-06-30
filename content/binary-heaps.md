title: Heaps for incremental computation
date: 2016-11-21
comments: true
tags: sampling, datastructures, incremental-computation


In this post, I'll describe a neat trick for maintaining a summary quantity
(e.g., sum, product, max, log-sum-exp, concatenation, cross-product) under
changes to its inputs. The trick and it's implementation are inspired by the
well-known max-heap datastructure. I'll also describe a really elegant
application to fast sampling under an evolving categorical distribution.


**Setup**: Suppose we'd like to efficiently compute a summary quantity under
changes to its $n$-dimensional input vector $\boldsymbol{w}$. The particular
form of the quantity we're going to compute is $z = \bigoplus_{i=1}^n w_i$,
where $\oplus$ is some associative binary operator with identity element
$\boldsymbol{0}$.

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

<button class="toggle-button" onclick="toggle('#operator-mathy');">more formally...</button>
<div id="operator-mathy" class="derivation" style="display:none">

* $\boldsymbol{w} \in \boldsymbol{K}^n$

* $\oplus: \boldsymbol{K} \times \boldsymbol{K} \mapsto \boldsymbol{K}$.

* Associative: $(a \oplus b) \oplus c = a \oplus (b \oplus c)$ for all $a,b,c
  \in \boldsymbol{K}$.

* Identity element: $\boldsymbol{0} \in \boldsymbol{K}$ such that $k \oplus
  \boldsymbol{0} = \boldsymbol{0} \oplus k = k$, for all $k \in \boldsymbol{K}$.

</div>

**The trick**: Essentially, the trick boils down to *parenthesis placement* in
the expression which computes $z$. A freedom we assumed via the associative
property.

I'll demonstrate by example with $n=8$.

Linear structure: We generally compute something like $z$ with a simple
loop. This looks like a right-branching binary tree when we think about the
order of operations,

$$
z = (((((((w_1 \oplus w_2) \oplus w_3) \oplus w_4) \oplus w_5) \oplus w_6) \oplus w_7) \oplus w_8).
$$

<br/> Heap structure: Here the parentheses form a balanced tree, which looks
much more like a recursive implementation that computes the left and right
halves and $\oplus$s the results (divide-and-conquer style),

$$
z = (((w_1 \oplus w_2) \oplus (w_3 \oplus w_4)) \oplus ((w_5 \oplus w_6) \oplus (w_7 \oplus w_8))).
$$

<br/>
The benefit of the heap structure is that there are $\mathcal{O}(\log n)$
intermediate quantities that depend on any input, whereas the linear structure
has $\mathcal{O}(n)$. The intermediate quantities correspond to the values of each of the
parenthesized expressions.

Since fewer intermediate quantities depend on a given input, fewer intermediates
need to be adjusted upon a change to the input. Therefore, we get faster
algorithms for *maintaining* the output quantity $z$ as the inputs change.

**Heap datastructure** (aka
[binary index tree or Fenwick tree](https://en.wikipedia.org/wiki/Fenwick_tree)):
We're going to store the values of the intermediates quantities and inputs in a
heap datastructure, which is a *complete* binary tree. In our case, the tree has
depth $1 + \lceil \log_2 n \rceil$, with the values of $\boldsymbol{w}$ at it's
leaves (aligned left) and padding with $\boldsymbol{0}$ for remaining
leaves. Thus, the array's length is $< 4 n$.

This structure makes our implementation really nice and efficient because we
don't need pointers to find the parent or children of a node (i.e., no need to
wrap elements into a "node" class like in a general tree data structure). So, we
can pack everything into an array, which means our implementation has great
memory/cache locality and low storage overhead.

Traversing the tree is pretty simple: Let $d$ be the number of internal nodes,
nodes $1 \le i \le d$ are internal. For node $i$, left child $\rightarrow {2
\cdot i},$ right child $\rightarrow {2 \cdot i + 1},$ parent $\rightarrow
\lfloor i / 2 \rfloor.$ (Note that these operations assume the array's indices
start at $1$. We generally fake this by adding a dummy node at position $0$,
which makes implementation simpler.)

**Initializing the heap**: Here's code that initializes the heap structure we
  just described.

```python
def sumheap(w):
    "Create sumheap from weights `w` in O(n) time."
    n = w.shape[0]
    d = int(2**np.ceil(np.log2(n)))  # number of intermediates
    S = np.zeros(2*d)                # intermediates + leaves
    S[d:d+n] = w                     # store `w` at leaves.
    for i in reversed(range(1, d)):
        S[i] = S[2*i] + S[2*i + 1]
    return S
```

**Updating $w_k$** boils down to fixing intermediate sums that (transitively)
  depend on $w_k.$ I won't go into all of the details here, instead I'll give
  code (below). I'd like to quickly point out that the term "parents" is not
  great for our purposes because they are actually the *dependents*: when an
  input changes the value the parents, grand parents, great grand parents, etc,
  become stale and need to be recomputed bottom up (from the leaves). The code
  below implements the update method for changing the value of $w_k$ and runs in
  $\mathcal{O}(\log n)$ time.


```python
def update(S, k, v):
    "Update w[k] = v` in time O(log n)."
    d = S.shape[0]
    i = d//2 + k
    S[i] = v
    while i > 0:   # fix parents in the tree.
        i //= 2
        S[i] = S[2*i] + S[2*i + 1]
```

Remarks
-------

 * **Numerical stability**: If the operations are noisy (e.g., floating point
   operator), then the heap version may be better behaved. For example, if
   operations have an independent, additive noise rate $\varepsilon$ then noise
   of $z_{\text{heap}}$ is $\mathcal{O}(\varepsilon \cdot \log n)$, whereas
   $z_{\text{linear}}$ is $\mathcal{O}(\varepsilon \cdot n)$. (Without further
   assumptions about the underlying operator, I don't believe you can do better
   than that.)

 * **Relationship to max-heap**: In the case of a max or min heap, we can avoid
   allocating extra space for intermediate quantities because all intermediates
   values are equal to exactly one element of $\boldsymbol{w}$.

 * **Change propagation**: The general idea of *adjusting* cached intermediate
   quantities is a neat idea. In fact, we encounter it each time we type
   ``make`` at the command line! The general technique goes by many
   names&mdash;including change propagation, incremental maintenance, and
   functional reactive programming&mdash;and applies to basically *any*
   side-effect-free computation. However, it's most effective when the
   dependency structure of the computation is sparse and requires little
   overhead to find and refresh stale values. In our example of computing $z$,
   these considerations manifest themselves as the heap vs linear structures and
   our fast array implementation instead of a generic tree datastructure.


Generalizations
---------------

 * No zero? No problem. We don't *actually* require a zero element. So, it's
   fair to augment $\boldsymbol{K} \cup \{ \textsf{null} \}$ where
   $\textsf{null}$ is distinguished value (i.e., $\textsf{null} \notin
   \boldsymbol{K}$) that *acts* just like a zero after we overload $\oplus$ to
   satisfy the definition of a zero (e.g., by adding an if-statement).

 * Generalization to an arbitrary maps instead of fixed vectors is possible with
   a "locator" map, which a bijective map from elements to indices in a dense
   array.

 * Support for growing and shrinking: We support **growing** by maintaining an
   underlying array that is always slightly larger than we need&mdash;which
   we're *already* doing in the heap datastructure. Doubling the size of the
   underlying array (i.e., rounding up to the next power of two) has the added
   benefit of allowing us to grow $\boldsymbol{w}$ at no asymptotic cost!  This
   is because the resize operation, which requires an $\mathcal{O}(n)$ time to
   allocate a new array and copying old values, happens so infrequently that
   they can be completely amortized. We get of effect of **shrinking** by
   replacing the old value with $\textsf{null}$ (or $\boldsymbol{0}$). We can
   shrink the underlying array when the fraction of nonzeros dips below
   $25\%$. This prevents "thrashing" between shrinking and growing.


Application
-----------

**Sampling from an evolving distribution**: Suppose that $\boldsymbol{w}$
corresponds to a categorical distributions over $\{1, \ldots, n\}$ and that we'd
like to sample elements from in proportion to this (unnormalized) distribution.

Other methods like the [alias](http://www.keithschwarz.com/darts-dice-coins/) or
inverse CDF methods are efficient after a somewhat costly initialization
step. But! they are not as efficient as the heap sampler when the distribution
is being updated. (I'm not sure about whether variants of alias that support
updates exist.)

<center>

  | Method |  Sample  |  Update  | Init |
  | ------ | -------- | -------- | ---- |
  |  alias |   O(1)   |  O(n)?   | O(n) |
  |  i-CDF | O(log n) |  O(n)    | O(n) |
  |  heap  | O(log n) | O(log n) | O(n) |

</center>

Use cases include

* [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), where
  distributions are constantly modified and sampled from (changes may not be
  sparse so YMMV). The heap sampler is used in
  [this paper](https://arxiv.org/abs/1412.4986).

* [EXP3](https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/)
  ([mutli-armed bandit algorithm](https://en.wikipedia.org/wiki/Multi-armed_bandit))
  is an excellent example of an algorithm that samples and modifies a single
  weight in the distribution.

* *Stochastic priority queues* where we sample proportional to priority and the
  weights on items in the queue may change, elements are possibly removed after
  they are sampled (i.e., sampling without replacement), and elements are added.

Again, I won't spell out all of the details of these algorithms. Instead, I'll
just give the code.

**Inverse CDF sampling**

```python
def sample(w):
    "Ordinary sampling method, O(n) init, O(log n) per sample."
    c = w.cumsum()            # build cdf, O(n)
    p = uniform() * c[-1]     # random probe, p ~ Uniform(0, z)
    return c.searchsorted(p)  # binary search, O(log n)
```

**Heap sampling** is essentially the same, except the cdf is stored as heap,
which is perfect for binary search!

```python
def hsample(S):
    "Sample from sumheap, O(log n) per sample."
    d = S.shape[0]//2     # number of internal nodes.
    p = uniform() * S[1]  # random probe, p ~ Uniform(0, z)
    # Use binary search to find the index of the largest CDF (represented as a
    # heap) value that is less than a random probe.
    i = 1
    while i < d:
        # Determine if the value is in the left or right subtree.
        i *= 2         # Point at left child
        left = S[i]    # Probability mass under left subtree.
        if p > left:   # Value is in right subtree.
            p -= left  # Subtract mass from left subtree
            i += 1     # Point at right child
    return i - d
```

**Code**: Complete code and test cases for heap sampling are available in this
[gist](https://gist.github.com/timvieira/da31b56436045a3122f5adf5aafec515).
