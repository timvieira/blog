title: From Weighted Grammars to Dyna Programs
date: 2026-11-02
tags: semirings, grammars, dyna, dynamic-programming, programming-languages
status: draft

<macros>
\renewcommand{\otimes}{\mathbin{\circ}}
\newcommand{\Values}{{\color{magenta}{\mathbb{V}}}}
\newcommand{\semione}{{\color{magenta}{\underline{1}}}}
\newcommand{\semizero}{{\color{magenta}{\underline{0}}}}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
\newcommand{\sem}[1]{{\color{magenta}{\llbracket}}#1{\color{magenta}{\rrbracket}}}
\newcommand{\w}{{\color{magenta}{w}}}
</macros>

<style>
.theorem { border-left: 3px solid #c9c2b8; padding: 0.6em 1em; margin: 1.2em 0; background: #fbf8f3; }
.theorem > :first-child { margin-top: 0; }
.theorem > :last-child { margin-bottom: 0; }
/* notebook outputs: wide graphs and long lines scroll sideways at full size */
.jp-OutputArea-output { overflow-x: auto; max-width: 100%; }
.jp-OutputArea-output svg { max-width: none; }
.jp-OutputArea-output pre { white-space: pre; overflow-x: auto; }
</style>

<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 1.5em; border-radius: 4px; font-size: 0.95em;"><p style="margin: 0;"><strong>Draft</strong> — This article is a work in progress. Content may change.</p></div>

My series on solving weighted grammars ([part 1](https://timvieira.github.io/semiring-newton/weighted-grammars-as-algebraic-systems/), [part 2](https://timvieira.github.io/semiring-newton/solving-linear-grammars-exactly/), [part 3](https://timvieira.github.io/semiring-newton/newtons-method-for-weighted-grammars/)) specialized chapter 5 of [my dissertation](https://timvieira.github.io/doc/2023-timv-dissertation.pdf) from *Dyna programs* to weighted context-free grammars.  This post goes the other way: it introduces Dyna and shows that a weighted grammar is a special case of a Dyna program, so everything in the series (fixpoint iteration, Newton's method, the linear solver, and the hybrid solver) carries over.


## Dyna

Dyna is a high-level, domain-specific language for concisely expressing dynamic programs as recurrence relations.  It was first introduced by [Eisner, Goldlust, and Smith (2005)](https://aclanthology.org/H05-1036/), where it was demonstrated to be a useful notation and software library for rapidly developing efficient algorithms for a variety of popular models of linguistic structure (such as tagging, parsing, and translation).

<div class="theorem" markdown="1">
**Example.** The total weight of length-4 paths in a weighted directed graph:

```text
total += w(Y1,Y2) * w(Y2,Y3) * w(Y3,Y4) * w(Y4,Y5).
```
</div>

This program defines the **value** of a derived **item** `total` in terms of input items `w(...)`.  The program assumes that we have encoded the set of edges in the given graph by declaring values for `w(...)` items.

Dyna rules may be recursive:

<div class="theorem" markdown="1">
**Example.** Minimum-cost path in a graph.

```text
goal min= start(I) + beta(I).
beta(J) min= stop(J).
beta(I) min= w(I,J) + beta(J).
```
</div>

This program defines `beta(I)` as the cost of the minimum-cost path in a graph from the node `I` to any terminal node `J`.  Here `stop(J)` must first be defined to be the cost of stopping at `J` when `J` is a terminal node (and should be left undefined otherwise).  Also, `w(I,J)` must first be defined to be the cost of the directed edge from `I` to `J`.

**Separation of logic and control.**  This program specifies the problem of finding the single-target shortest path in a weighted directed graph.  However, the details of the implementation are abstracted away.  Thus, many well-known single-target shortest-path algorithms, such as the Bellman–Ford–Moore algorithm and Dijkstra's algorithm, are represented as the same Dyna program, as they are based on the same recursive factoring of paths.

Notice that if we change the operations from $\langle \min, + \rangle$ to $\langle \max, \times \rangle$, we get the Viterbi algorithm for finding the most probable path in a graph from nodes in a set `start` to nodes in a set `stop`, where the probability of a path is given by the product of edge probabilities:

<div class="theorem" markdown="1">
**Example.** Finding the most probable path in a graph.

```text
goal max= start(J) * beta(J).
beta(J) max= stop(J).
beta(J) max= beta(I) * w(I,J).
```
</div>

What is interesting about the connection between these two programs is that we kept the same recurrence, but we have changed the aggregate quantity that we seek to compute.  Both programs define the same set of derivations, which we can interpret as paths in the graph.  How these paths are weighted and interpreted differs: this is the separation of concerns that [semirings](/blog/semirings-a-powerful-abstraction-for-dynamic-programming/) provide.

Next, we give a fundamental algorithm in NLP.

<div class="theorem" markdown="1">
**Example.** Weighted context-free parsing with CKY, or, more precisely, the inside algorithm:

```text
beta(X,I,K) += gamma(X,Y,Z) * beta(Y,I,J) * beta(Z,J,K).
beta(X,I,K) += gamma(X,Y) * beta(Y,I,K).
beta(X,I,K) += gamma(X,Y) * word(Y,I,K).
goal += beta(s,0,N) * len(N).
```
</div>

The values of the `gamma` items should be defined to be the weights of the corresponding context-free grammar rules: for example, the item `gamma(s,np,vp)` $= 0.7$ encodes the production $\mathtt{s} \xrightarrow{0.7} \mathtt{np}\; \mathtt{vp}$.  Also, the item `word(X,I,K)` should be $\semione$ if the input word `X` appears at position `K` of the input sentence and `I` $=$ `K` $-1$, and should be left undefined otherwise.  Then, for any nonterminal symbol `X` and any substring spanning positions $(I, K]$ of the input sentence, the item `beta(X,I,K)` represents the total weight of all derivations of that substring from `X`.

**Syntax and terminology.**  A Dyna program serves to define **values** for **items**; we denote the value of an item `x` as $\sem{\mathtt{x}}$.  Rules combine values with operations $\oplus$ and $\otimes$ that form a semiring, so `x += y * z` means $\sem{\mathtt{x}} \mathrel{\oplus}= \sem{\mathtt{y}} \otimes \sem{\mathtt{z}}$.  If an item's value is left undefined or has no $\oplus$-summands, this is equivalent to saying it has value $\semizero$, which means it cannot affect the values of any other items; in effect, it does not exist.  Rules may contain **variables** (capitalized, such as `X`), which are universally quantified, so one rule stands for all of its **groundings**.


## Grammars are Dyna programs

The connection between context-free grammars and Dyna programs is direct.  A grammar is a Dyna program without variables:

| grammar | Dyna |
|:--------|:-----|
| terminal | value |
| nonterminal | ground item |
| rule $X \xrightarrow{\w} Y\, Z$ | `x += w * y * z.` |
| parse tree | derivation |
| the weight (yield) of a parse tree | the value of a derivation |
| the total weight of the derivations of a nonterminal, $\sem{X}$ | the total value of all derivations of an item, $\sem{\mathtt{x}}$ |

The primary difference is that Dyna does not restrict the set of terminals and nonterminals to finite sets: rules with variables, like the CKY program above, describe infinitely many ground rules at once.  The value of a Dyna program is defined exactly as for a grammar, as the sum over derivations, which is the least solution of the program's fixpoint equations when that sum is well-defined.  Anecdotally, exploring the connection between Dyna programs and context-free grammars is how we discovered the generalized left-corner transformation and the Newton solver in my dissertation.  Conversely, program transformations for Dyna may be useful as grammar transformations.

**What carries over.**  Every algorithm in the grammar series was originally stated for Dyna programs in chapter 5 of my dissertation: fixpoint iteration (naive, semi-naive, and prioritized), Newton's method, the right-linear solver, and the hybrid solver that breaks a program into strongly connected components.  The grammar versions are what you get by reading "program" as "grammar" and "item" as "nonterminal."  The main extra wrinkle for programs is that rules have variables, so the solvers must work with the program's groundings (or with nonground reasoning).

**Try it.**  The [`dyna-pi`](https://github.com/timvieira/dyna-pi) library implements these ideas.  The rest of this post is a tour adapted from its [tutorial notebook](https://github.com/timvieira/dyna-pi/blob/main/docs/Demo.ipynb).


## A tour of `dyna-pi`

The **dynamic program improvement system** (`dyna-pi`) is an interactive tool for analyzing and improving dynamic programming algorithms.  It provides the Dyna language, general-purpose solvers (agenda-based fixpoint iteration and semiring Newton's method), static analyses (types, and worst-case time and space complexity), meaning-preserving program transformations, a search algorithm over those transformations, and visualization tools that integrate with Jupyter.

{% notebook dyna-tutorial.ipynb cells[8:9] %}

### CKY

Here is the CKY program from above, written for `dyna-pi`.  The last two lines declare which items are inputs and which are outputs.

{% notebook dyna-tutorial.ipynb cells[14:24] %}

### Computation graphs and derivations

{% notebook dyna-tutorial.ipynb cells[25:30] %}

### Solvers

{% notebook dyna-tutorial.ipynb cells[33:35] %}

{% notebook dyna-tutorial.ipynb cells[38:39] %}

**What about cyclical programs?**  This is where the [grammar series](https://timvieira.github.io/semiring-newton/weighted-grammars-as-algebraic-systems/) picks up: cyclical programs define infinitely many derivations, so fixpoint iteration only converges in the limit.

{% notebook dyna-tutorial.ipynb cells[41:43] %}

{% notebook dyna-tutorial.ipynb cells[44:61] %}

### Semirings

The same program can be evaluated in different [semirings](/blog/semirings-a-powerful-abstraction-for-dynamic-programming/) without changing its rules.

{% notebook dyna-tutorial.ipynb cells[129:141] %}

### Program transformations

**Unfold and fold.**  Transformations rewrite a program into a semantically equivalent one that may be more efficient to run.

{% notebook dyna-tutorial.ipynb cells[63:65] %}

{% notebook dyna-tutorial.ipynb cells[66:74] %}

**Speeding up CKY.**  Folding in a new item (the "hook trick") changes the program's worst-case runtime, which `dyna-pi` can compute from a description of the input types:

{% notebook dyna-tutorial.ipynb cells[79:85] %}

Unfolding can also specialize the program to a specific grammar:

{% notebook dyna-tutorial.ipynb cells[88:91] %}

**The left-corner transformation.**  This example shows how the left-corner transformation can be used to speed up the following sum over weighted Peano numbers.

{% notebook dyna-tutorial.ipynb cells[163:171] %}

The [tutorial notebook](https://github.com/timvieira/dyna-pi/blob/main/docs/Demo.ipynb) goes further, with program analysis, type-based program specialization, and semi-automated search over transformations.


## References

* Jason Eisner, Eric Goldlust, and Noah A. Smith.  [Compiling Comp Ling: Weighted Dynamic Programming and the Dyna Language](https://aclanthology.org/H05-1036/).  HLT-EMNLP 2005.
* Tim Vieira.  [Automating the Analysis and Improvement of Dynamic Programming Algorithms with Applications to Natural Language Processing](https://timvieira.github.io/doc/2023-timv-dissertation.pdf).  PhD thesis, 2023.
