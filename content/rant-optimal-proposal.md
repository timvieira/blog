title: The optimal proposal distribution is not p
date: 2016-05-28
comments: true
tags: math, statistics, sampling, importance-sampling

The following is a quick rant about
[importance sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/).

I've heard the following **incorrect** statement one too many times,

> We chose $q \approx p$ because $q=p$ is the "optimal" proposal distribution.

While it is certainly a good idea to pick $q$ to be as similar as possible to
$p$, it is by no means *optimal* because it is oblivious to $f$!

With importance sampling, it is possible to achieve a variance reduction over
Monte Carlo estimation. The optimal proposal distribution, assuming $f(x) \ge 0$
for all $x$, is $q(x) \propto p(x) f(x).$ This choice of $q$ gives us a *zero
variance* estimate *with a single sample*!

Of course, this is an unreasonable distribution to use because the normalizing
constant *is the thing you are trying to estimate*, but it is proof that *better
proposal distributions exist*.

The key to doing better than $q=p$ is to take $f$ into account. Look up
"importance sampling for variance reduction" to learn more.
