title: Empirical Bernstein
date: 2014-12-28
comments: true
status: draft
tags: machine-learning

A trending tool for developing algorithms and tightening bounds is the empirical
Bernstein inequality.

Essentially one can generate a new algorithms by searching for Hoeffding
inequality and using the empirical Bernstein inequality.

The benefit of EB over Hoeffding is that the bound accounts for observed
variance, not just the range.

Example usage: high-confidence policy evaluation (cite: both papers)

TODO: plot both bounds on some sample data.
