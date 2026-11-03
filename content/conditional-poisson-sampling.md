title: Conditional Poisson Sampling
date: 2026-10-28
tags: notebook, sampling, algorithms, sampling-without-replacement
external_url: https://timvieira.github.io/conditional-poisson-sampling/

Suppose you want to draw a random subset of exactly $n$ items from a universe of $N$ items, where each item $i$ has a positive weight $w_i$.
The **conditional Poisson distribution** assigns each size-$n$ subset a probability proportional to the product of its weights:

$$
p(S) \propto \mathbf{1}\big[ |S| = n\big] \prod_{i \in S} w_i
$$

The normalizing constant $\binom{\boldsymbol{w}}{n} \overset{\small\text{def}}{=} \sum_{|S|=n} \prod_{i \in S} w_i$ is a weighted generalization of the binomial coefficient, which recovers $\binom{N}{n} = \binom{\boldsymbol{w}}{n}$ when $\boldsymbol{w} = \mathbf{1}^N$.
The inclusion probability is $\pi_i \overset{\small\text{def}}{=} \sum_{S} p(S)\, \mathbf{1}[i \in S]$.

**What this post covers.** The computational challenges are: computing $\binom{\boldsymbol{w}}{n}$ and $p(S)$, computing $\boldsymbol{\pi}$ from $\boldsymbol{w}$, drawing exact samples $S \sim p$, and the inverse problem of finding $\boldsymbol{w}$ from target $\boldsymbol{\pi}$.  This post gives efficient algorithms for all four—in $\mathcal{O}(N \log^2 n)$ time using a polynomial product tree—and times them against the $\mathcal{O}(Nn)$ dynamic program used by existing software.

**Software.** The code is available as a [Python library](https://github.com/timvieira/conditional-poisson-sampling).  As far as I can tell, this is the only publicly available library for conditional Poisson sampling in Python (or any language outside of R's survey-sampling packages).
