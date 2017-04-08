title: How to test gradient computation
date: 2017-04-08
comments: true
status: draft
tags: testing, calculus


**Who sould read this?** Nowadays, you're probably just using automatic
differentiation to compute the gradient of whatever function you're using. If
that's you and you trust your software package wholeheartedly, then you probably
don't need to read this. If you're rolling your own module/Op to pluf in to an
auto-diff library, then you should read this. I find that none of the available
libraries are any good for differentiating dynamic programs, so I still find
this stuff useful. You'll probably have to write gradient code without the aid
of an autodiff library someday... So either way, knowing this stuff is good for
you. To answer the question, everyone.

**The setup**: Suppose we have a function, $f: \mathbb{R}^d \rightarrow
\mathbb{R}$, and we want to test code that computes $\nabla f$. Note that these
techniques also apply when $f$ has multi-variate output; I'll briefly touch on
that in the section "extra credit 1".

The main way that people test gradient computation is by the finite-difference
gradient test.

Before we get into that, read this disclaimer:

Rule #0: Always test the function. Even if the gradient is correct, it might be
the gradient of the wrong function! How you test the function depends strongly
on what it's supposed to compute.

 - Example: for a log-linear model you can also test that $\log Z_\theta(x)$ is
   computed correctly by brute-force enumeration of $\mathcal{Y}(x)$ on small
   examples.

You can directly test the gradient if you know a different way to compute it.

 - Example: For log-linear models, we know that the $\nabla \log Z_\theta(x)$
   computes a feature expection, which you can also compute by brute-force
   enumeration on small enough examples.


The finite-difference approximation to the gradient
([further reading](http://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/))
is a good way to compute a gradient in a different way that whatever you
implemented in your code. The
[complex-step method](http://timvieira.github.io/blog/post/2014/08/07/complex-step-derivative/)
is also an option if you're code supports complex numbers.

Rule #1: Always use two-sided difference!

 - Remark 1.1. For low-dimensional functions, you can straight-up use the
   finite-difference approximation instead of rolling code to compute the
   gradient.

   This numerical approximation is in fact very accurate on almost any smooth
   function; the numerical stability of finite-difference is well studied and
   you can read about it in any introductory text on numerical methods. Of
   course, specialized code is probably a little more accurate.

 - Remark 1.2. The reason why we write specialized gradient code is *not*
   improve numerical accuracy, it's to improve efficiency.

   As I've
   [ranted](http://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)
   before, automatic differentiation techniques guarantee that evaluating
   $\nabla f(x)$ gradient is as computationally efficient as computing $f(x)$
   with the caveat that *space* complexity may increase substantially.

   FD is $\mathcal{O}(d \cdot \textrm{runtime } f(x))$, where as autodiff is
   $\mathcal{O}(\textrm{runtime } f(x))$


Rule #2: Test multiple inputs (values of parameters, training examples, dropout
vectors, etc.)

Rule #3: Absolute difference is the devil - vectors/scalars do not have a common
scale (unless you have some special structure) so you should never never never
compare in absolute difference (this is Lecture 1 of any numerical methods
course).

Most people use relative error to put things in a common scale, but
unfortunately relative error chokes on zeros.

I compute several error measures with a script that you can inport from my
github
[arsenal.math.checkgrad.{fdcheck}](https://github.com/timvieira/arsenal/blob/master/arsenal/math/checkgrad.py)

I use several metrics.
1. relative error (annoying for zeros),
2. absolute error
3. cosine similarity and/or Pearson correlation

   - Cosine ignores scale. Pearson ignores scale and shift.

   - No prob with zeros (unless all are zero)

   - Checks the *direction*, often code is off by a constant factor - this in
     conjunction with the other tests will help you catch that.

   - Note: Cosine has to be really high, like 99.9999.

   - Rule #3.a Always fix the scale errors! (e.g. In the log-linear model
     example, you might have forgotten to divide by $Z$, so it's not really a
     constant and your optimization won't work when try to use it on multiple
     examples.)

4. Accuracy of predicting the sign {+,-,0} of each dimension.

I like to scatter plot the finite-difference vs the version in my when
debugging.

All these measurements and the plot are computed with
[arsenal.math.compare.{compare}](https://github.com/timvieira/arsenal/blob/master/arsenal/math/compare.py)


When debugging, I tend to use axis-aligned perturbation vectors because they
tell me something about the dimensions that are wrong, unlike a random
perturbation. The random perturbations are useful for when you are confident in
your code and want the test cases to run quickly. In that case, you can switch
to check a few random (use arsenal.math.spherical) directions using the general
gradient-vector product trick (i.e., non-axis-aligned vector) and you can then
insert assertions on thresholds of the different metrics.

Extra credit 1: You can test the different modules of you code as well. E.g., I
test my DP algorithm independent of how the features and downstream loss are
computed. You can also test feature and downstream loss modules independent of
one another. Note that autodiff implicitly module computes Jacobian-vector
product because modules are multivariate. For example, the inside algorithm
computes an entire table of betas, which you can test. The adjoints of the
output are the "vector" in question.

You can just take a random adjoint vector and use it reduce to the scalar
function case.

Something like this:
```python
r = spherical(D)
h = lambda x: module.fprop(x).dot(r)   # scalar function for use in fd

module.fprop(x)   # propagate
module.output.adjoint = r. # init adjoint to r, usually we set adjoint of scalar output=1
module.bprop()
ad = module.input.adjoint # grab the gradient
fd = fdgrad(h)
compare(fd, ad)
```

Extra credit 2: Another useful test for machine learning applications is to make
sure that you can overfit small datasets. This is a good
[integration test](https://en.wikipedia.org/wiki/Integration_testing), but
doesn't always apply.


Other common sources of bugs

Rule #4: Test that batch = minibatch (if you code does minibatching). It's very
easy to get this bit wrong. Broadcasting rules make it easy to hide matrix
conformability errors.

Rule #5: Really look over your test cases. I often find the my errors are
actually in the test case themselves because either (1) I wrote it really
quickly with less care than the difficult function/gradient, or (2) there is a
gap between "what I want it to do" and "what I told it to do".

Rule #6: Random search in the space of programs can result in overfitting! This
is a general problem with test-driven development that always applies. If you
are hamfistedly twiddling bits of your code without thinking about why things
work, you can trick almost any test.
