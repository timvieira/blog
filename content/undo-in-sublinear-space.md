title: Reversing a sequence with sublinear space
date: 2016-10-01
comments: true
tags: algorithms

Suppose we have a computation which generates sequence of states $s_1 \ldots
s_n$ according to $s_{t} = f(s_{t-1})$ where $s_0$ is given.

We'd like to devise an algorithm, which can reconstruct each point in the
sequence efficiently as we traverse it backwards. You can think of this as
"hitting undo" from the end of the sequence or reversing a singly-liked list.

Obviously, we *could* just record the entire sequence, but if $n$ is large *or*
the size of each state is large, this will be infeasible.

**Idea 0**: Rerun the forward pass $n$ times. Runtime $\mathcal{O}(n^2)$, space
  $\mathcal{O}(1)$.

**Idea 1**: Suppose we save $0 < k \le n$ evenly spaced "checkpoint" states.
Clearly, this gives us $\mathcal{O}(k)$ space, but what does it do to the
runtime?  Well, if we are at time $t$ the we have to "replay" computation from
the last recorded checkpoint to get $s_t$, which takes $O(n/k)$ time. Thus, the
overall runtimes becomes $O(n^2/k)$. This runtime is not ideal.

**Idea 2**: *Idea 1* did something kind of silly, within a chunk of size $n/k$,
it does each computation multiple times! Suppose we increase the memory
requirement *just a little bit* to remember the current chunk we're working on,
making it now $\mathcal{O}(k + n/k)$. Now, we compute each state at most $2$
times: once in the initial sequence and once in the reverse. This implies a
*linear* runtime.  Now, the question: how should we set $k$ so that we minimize
extra space? Easy! Solve the following little optimization problem:

$$
\underset{k}{\textrm{argmin}}\ k+n/k = \sqrt{n}
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

<button onclick="toggle('#derivation-optimal-space')" class="toggle-button">Derivation</button>
<div id="derivation-optimal-space" style="display:none;" class="derivation">
To get the minimum, we solve for $k$ that sets the derivative to zero.
$$
\begin{eqnarray}
    0 &=& \frac{\partial}{\partial k} \left[ k+n/k \right] \\
      &=& 1-n/k^2 \\
n/k^2 &=& 1 \\
  k^2 &=& n \\
    k &=& \sqrt{n}
\end{eqnarray}
$$
<br/>

Since it's safe to assume that $n,k \ge 1$ and $\frac{\partial^2}{\partial k\,
\partial k} = 2 n / k^3 > 0$ this is indeed a minimum. It's also global minimum
because $k+n/k$ is convex in $k$ when $n,k > 0$.

</div>

That's nuts! We get away with *sublinear* space $\mathcal{O}(\sqrt{n})$ and we
only blow up our runtime by a factor of 2. Also, I really love the "introduce a
parameter then optimize it out" trick.

<button onclick="toggle('#code-sqrt-space')">Code</button>
<div id="code-sqrt-space" style="display:none;">
```python
def sqrt_space(f, s0, n):
    k = int(ceil(sqrt(n)))
    memory = {}
    s = s0
    t = 0
    while t <= n:
        if t % k == 0:
            memory[t] = s
        s = f(s)
        t += 1
    b = n
    while b >= k:
        # last chunk may be shorter than k.
        c = ((n % k) or k) if b == n else k
        for s in reversed(step(f, memory[b-c], c)):
            yield s
            b -= 1

def step(f, s, k):
    "Take `k` steps from state `s`, save path. Cost: O(k) space, O(k) time."
    if k == 0:
        return []
    B = [s]
    for _ in range(k-1):
        s = f(s)
        B.append(s)
    return B
```
</div>

**Idea 3**: What if we apply "the remember $k$ states" trick *recursively*? I'm
going to work this out for $k=2$ (and then claim that the value of $k$ doesn't
matter).

Run forward to get the midpoint at $s_{m}$, where $m=b + \lfloor n/2
\rfloor$. Next, recurse on the left and right chunks $[b,m)$ and $[m,e)$.
We hit the base case when the width of the interval is
one.

Note that we implicitly store midpoints as we recurse (thanks to the stack
frame).  The max depth of the recursion is $\mathcal{O}(\log n)$, which gives us
a $\mathcal{O}(\log n)$ space bound.

We can characterize runtime with the following recurrence relation, $T(n) = 2
\cdot T(n/2) + \mathcal{O}(n)$. Since we recognize this as the recurrence for
mergesort, we know that it flattens to $\mathcal{O}(n \log n)$ time. Also, just
like in the case of sorting, the branching factor doesn't matter so we're happy
with or initial assumption that $k=2$.

<button onclick="toggle('#code-recursive')">Code</button>
<div id="code-recursive" style="display:none;">
```python
def recursive(f, s0, b, e):
    if e - b == 1:
        yield s0
    else:
        # do O(n/2) work to find the midpoint with O(1) space.
        s = s0
        d = (e-b)//2
        for _ in range(d):
            s = f(s)
        for s in recursive(f, s, b+d, e):
            yield s
        for s in recursive(f, s0, b, b+d):
            yield s
```
</div>


## Remarks

The algorithms describe in this post are generic algorithmic tricks, which has
been used in a number of place, including

* The classic computer science interview problem of reversing a singly-linked list
  under a tight budget on *additional* memory.

* Backpropagation for computing gradients in sequence models, including HMMs ([Zweig & Padmanabhan, 2000](https://www.microsoft.com/en-us/research/wp-content/uploads/2000/01/icslp00_logspace.pdf))
  and RNNs ([Chen et al., 2016](https://arxiv.org/abs/1604.06174v2)). I have
  sample code that illustrates the basic idea below.

* Memory-efficient [omniscient debugging](https://arxiv.org/pdf/cs/0310016v1),
  which allows a user to inspect program state while moving forward *and
  backward* in time.

## Sample code

* [The basics](https://gist.github.com/timvieira/d2ac72ec3af7972d2471035011cbf1e2):
  Simple implementation complete with test cases.

* [Memory-efficient backprop in an RNN](https://gist.github.com/timvieira/aceb64047aed1b13bf4e4da3b9a4c0e):
  A simple application with test cases, of course.
