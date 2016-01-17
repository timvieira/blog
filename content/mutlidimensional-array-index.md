title: Multidimensional array index
date: 2016-01-17
comments: true
tags: misc


This is a simple note on how to compute the bijective mapping between the
indices of an $n$-dimensional array and a flat, one-dimensional array. We'll
look at both directions of the mapping: ``(tuple->int)`` and ``(int -> tuple)``.

This mapping can be used as a perfect hash for tuples, assuming each dimension
$a, b, c, \ldots$ is a positive integer and bounded $a \le A, b \le B, c \le C,
\ldots$


### Start small

Let's start by looking at $n = 3$ and generalize from there.

```python
def index_3(a, A):
    _,J,K = A
    i,j,k = a
    return ((i*J + j)*K + k)

def inverse_3(ix, A):
    _,J,K = A
    total = J*K
    i = ix // total
    ix = ix % total
    total = K
    j = ix // total
    k = ix % total
    return (i,j,k)
```

Here's our test case:

```python
A,B,C = 3,4,5
key = 0
for a in range(A):
    for b in range(B):
        for c in range(C):
            print (a,b,c), '->', key
            assert inverse_3(key, (A,B,C)) == (a,b,c)
            assert index_3((a,b,c), (A,B,C)) == key
            key += 1
```

Note: This is not the only bijective mapping from ``tuple`` to ``int`` that we
could have come up with. The one we chose corresponds to the particular layout,
which is apparent in the test case.

For $n=4$ the pattern is $((a \cdot B + b) \cdot C + d) \cdot D + d$.

Sidenote: We don't actually need the bound $a \le A$ in either ``index`` or
``inverse``. This gives us a little extra flexibility because our first
dimension can be infinite/unknown.

### General case

```python
def index(a, A):
    "Map tuple ``a`` to index with known bounds ``A``."
    # the pattern:
    # ((i*J + j)*K + k)*L + l
    key = a[0]
    for i in xrange(1, len(A)):
        key *= A[i]
        key += a[i]
    return key

def inverse(ix, A):
    "Find key given index ``ix`` and bounds ``A``."
    total = 1
    for x in A:
        total *= x
    key = []
    for i in xrange(len(A)):
        total /= A[i]
        r = ix // total
        ix = ix % total
        key.append(r)
    return key
```


## Appendix

### Testing the general case

```python
import numpy as np, itertools

def test_layout(D):
    "Test that `index` produces the layout we expect."
    z = [index(d, D) for d in itertools.product(*(range(a) for a in D))]
    assert z == range(np.product(D))

def test_inverse(key, D):
    got = inverse(index(key, D), D)
    assert tuple(key) == tuple(got)

if __name__ == '__main__':
    test_layout([3,5,7,2])
    test_layout([3,5,7])
    test_layout([3,5])
    test_layout([3])

    test_inverse(key = (1,), D = (10,))
    test_inverse(key = (1,3), D = (2,4))
    test_inverse(key = (3,2,5), D = (10,4,8))
    test_inverse(key = (3,2,5,1), D = (10,4,8,2))
    test_inverse(key = (3,2,5,1,11), D = (10,4,8,2,20))
```
