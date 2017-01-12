Title: Numerically-stable p-norms
date: 2014-11-10
tags: numerical
comments: true

Consider the p-norm
$$
|| \boldsymbol{x} ||_p = \left( \sum_i |x_i|^p \right)^{\frac{1}{p}}
$$

In python this translates to:

```python
from numpy import array
    
def norm1(x, p):
    "First-pass implementation of p-norm."
    return (x**p).sum() ** (1./p)
```

Now, suppose $|x_i|^p$ causes overflow (for some $i$). This will occur for sufficiently large $p$ or sufficiently large $x_i$---even if $x_i$ is representable (i.e., not NaN or $\infty$).

For example:


```python
>>> big = 1e300
>>> x = array([big])
>>> norm1(x, p=2)
[ inf]   # expected: 1e+300
```

This fails because we can't square ``big``

```python
>>> np.array([big])**2
[ inf]
```


## A little math

There is a way to avoid overflowing because of a few large $x_i$.

Here's a little fact about p-norms: for any $p$ and $\boldsymbol{x}$
$$
|| \alpha \cdot \boldsymbol{x} ||_p = |\alpha| \cdot || \boldsymbol{x}   ||_p
$$

We'll use the following version (harder to remember)
$$
|| \boldsymbol{x} ||_p  = |\alpha| \cdot || \boldsymbol{x} / \alpha ||_p
$$

Don't believe it? Here's some algebra:
$$
\begin{eqnarray*}
|| \boldsymbol{x} ||_p
&=& \left( \sum_i |x_i|^p \right)^{\frac{1}{p}} \\
&=& \left( \sum_i \frac{|\alpha|^p}{|\alpha|^p} \cdot |x_i|^p \right)^{\frac{1}{p}} \\
&=& |\alpha| \cdot \left( \sum_i \left( \frac{|x_i| }{|\alpha|} \right)^p \right)^{\frac{1}{p}} \\
&=& |\alpha| \cdot \left( \sum_i \left| \frac{x_i }{\alpha} \right|^p \right)^{\frac{1}{p}} \\
&=& |\alpha| \cdot || \boldsymbol{x} / \alpha ||_p
\end{eqnarray*}
$$

## Back to numerical stability

Suppose we pick $\alpha = \max_i |x_i|$. Now, the largest number we have to take
the power of is one --- making it very difficult to overflow on the account of
$\boldsymbol{x}$. This should remind you of the infamous log-sum-exp trick.


```python
def robust_norm(x, p):
    a = np.abs(x).max()
    return a * norm1(x / a, p)
```

Now, our example from before works :-)

```python
>>> robust_norm(x, p=2)
1e+300
```

## Remarks


* It appears as if `scipy.linalg.norm` is robust to overflow, while `numpy.linalg.norm` is not. Note that `scipy.linalg.norm` appears to be a bit slower.

* The `logsumexp` trick is nearly identical, but operates in the log-domain, i.e., $\text{logsumexp}(\log(|x|) \cdot p) / p = \log || x ||_p$. You can implement both tricks with the same code, if you use different number classes for log-domain and real-domain---a trick you might have seen before.


```python
from arsenal.math import logsumexp
from numpy import log, exp, abs
>>> p = 2
>>> x = array([1,2,4,5])
>>> logsumexp(log(abs(x)) * p) / p
1.91432069824
>>> log(robust_norm(x, p))
1.91432069824
```


    
