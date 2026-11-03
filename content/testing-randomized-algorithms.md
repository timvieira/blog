title: Ideas for Testing Randomized Algorithms
date: 2026-11-01
tags: testing, sampling, algorithms, monte-carlo, program-tracing
external_url: https://timvieira.github.io/testing-randomized-algorithms/

Randomized algorithms are hard to test. Their outputs are, by design,
unpredictable&mdash;and that unpredictability is a *feature*, not a bug. It is what
gives them useful guarantees like unbiasedness and computational efficiency. But
it also makes them slippery: how do you check that a stochastic program is
correct when it gives different answers every time you run it?

This post sketches some ideas for how to think about testing randomized
algorithms&mdash;especially the kinds of algorithms that we use for probabilistic
inference. I'll present a concrete sampling algorithm with several plausible
implementations&mdash;some correct, some subtly broken&mdash;and develop
progressively more powerful testing strategies to tell them apart. Each
strategy catches a bug that the previous one missed.

The algorithm is a small building block. Its job is to propose the next token
inside sequential Monte Carlo (SMC) for constrained generation from language
models. Testing the block in isolation is only half the job, so at the end
we'll plug it into SMC and test the whole pipeline.
