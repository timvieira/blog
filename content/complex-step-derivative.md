title: Complex-step derivative
date: 2014-08-07
comments: true
tags: calculus

Estimate derivatives by simply passing in a complex number to your function!

$$
f'(x) \approx \frac{1}{\varepsilon} \text{Im}\Big[ f(x + i \cdot \varepsilon) \Big]
$$

<br/>Recall, the centered-difference approximation is a fairly accurate method for
approximating derivatives of a univariate function $f$, which only requires two
function evaluations. A similar derivation, based on the Taylor series expansion
with a complex perturbation, gives us a similarly-accurate approximation with a
single (complex) function evaluation instead of two (real-valued) function
evaluations. Note: $f$ must support complex inputs (in frameworks, such as numpy
or matlab, this often requires no modification to source code).

This post is based on
[Martins+'03](http://mdolab.engin.umich.edu/sites/default/files/Martins2003CSD.pdf).

**Derivation**: Start with the Taylor series approximation:

$$
f(x + i \cdot \varepsilon) =
  \frac{i^0 \varepsilon^0}{0!} f(x)
+ \frac{i^1 \varepsilon^1}{1!} f'(x)
+ \frac{i^2 \varepsilon^2}{2!} f''(x)
+ \frac{i^3 \varepsilon^3}{3!} f'''(x)
+ \cdots
$$

<br/>Take the imaginary part of both sides and solve for $f'(x)$. Note: the $f$ and
$f''$ term disappear because $i^0$ and $i^2$ are real-valued.

$$
f'(x) = \frac{1}{\varepsilon} \text{Im}\Big[ f(x + i \cdot \varepsilon) \Big] + \frac{\varepsilon^2}{3!} f'''(x) + \cdots
$$

<br/>As usual, using a small $\varepsilon$ let's us throw out higher-order
terms. And, we arrive at the following approximation:

$$
f'(x) \approx \frac{1}{\varepsilon} \text{Im}\Big[ f(x + i \cdot \varepsilon) \Big]
$$

<br/>If instead, we take the real part and solve for $f(x)$, we get an approximation
to the function's value at $x$:

$$
f(x) \approx \text{Re}\Big[ f(x + i \cdot \varepsilon) \Big]
$$

<br/>In other words, a single (complex) function evaluations computes both the
function's value and the derivative.

**Code**:
```python
def complex_step(f, eps=1e-10):
    """
    Higher-order function takes univariate function which computes a value and
    returns a function which returns value-derivative pair approximation.
    """
    def f1(x):
        y = f(complex(x, eps))         # convert input to complex number
        return (y.real, y.imag / eps)  # return function value and gradient
    return f1
```

A simple test:
```python
f = lambda x: exp(x)+cos(x)+10  # function
g = lambda x: exp(x)-sin(x)     # gradient
x = 1.0
print (f(x), g(x))
print complex_step(f)(x)
```


**Other comments**

- Using the complex-step method to estimate the gradients of multivariate
  functions requires independent approximations for each dimension of the
  input.

- Although the complex-step approximation only requires a single function
  evaluation, it's unlikely faster than performing two function evaluations
  because operations on complex numbers are generally much slower than on floats
  or doubles.


**Code**: Check out the
[gist](https://gist.github.com/timvieira/3d3db3e5e78e17cdd103) for this post.
