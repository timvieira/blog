title: Solving Weighted Grammars with Newton's Method
date: 2026-11-04
tags: semirings, grammars, parsing, algorithms, dynamic-programming
status: draft
external_url: https://timvieira.github.io/semiring-newton/

A weighted grammar's values are the least solution of a system of polynomial equations over a semiring.  Fixpoint iteration sums derivations by height and can be exponentially slow; Newton's method, generalized to semirings, sums them by dimension.  This post, adapted from chapter 5 of my dissertation, covers Newton's method, an exact solver for right-linear grammars, and a hybrid solver that breaks a grammar into strongly connected components, with a Python implementation for commutative semirings.
