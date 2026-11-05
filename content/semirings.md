title: Semirings: A Powerful Abstraction for Dynamic Programming
date: 2026-11-05
tags: semirings, algorithms, dynamic-programming
status: draft

<macros>
\renewcommand{\otimes}{\mathbin{\circ}}
\newcommand{\Values}{{\color{magenta}{\mathbb{V}}}}
\newcommand{\semione}{{\color{magenta}{\underline{1}}}}
\newcommand{\semizero}{{\color{magenta}{\underline{0}}}}
\newcommand{\va}{{\color{magenta}{a}}}
\newcommand{\vb}{{\color{magenta}{b}}}
\newcommand{\vc}{{\color{magenta}{c}}}
\newcommand{\vd}{{\color{magenta}{d}}}
\newcommand{\KleeneStar}{{\color{#332288}{\star}}}
\newcommand{\kleene}[1]{#1^{\KleeneStar}}
\newcommand{\dtrue}{\mathsf{true}}
\newcommand{\dfalse}{\mathsf{false}}
\newcommand{\dunk}{\mathsf{unk}}
\newcommand{\defeq}{\overset{\text{def}}{=}}
</macros>

<style>
.theorem { border-left: 3px solid #c9c2b8; padding: 0.6em 1em; margin: 1.2em 0; background: #fbf8f3; }
.theorem > :first-child { margin-top: 0; }
.theorem > :last-child { margin-bottom: 0; }
</style>

<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 1.5em; border-radius: 4px; font-size: 0.95em;"><p style="margin: 0;"><strong>Draft</strong> — This article is a work in progress. Content may change.</p></div>

Semirings are a powerful abstraction for dynamic programming.  They provide a useful and elegant abstraction level for

* algorithms for NLP (e.g., Teitelbaum, 1973; Mohri, 1997; [Goodman, 1999](https://aclanthology.org/J99-4004/); Huang, 2008)
* probabilistic inference (e.g., Aji and McEliece, 2000)
* database provenance (e.g., [Green et al., 2007](https://doi.org/10.1145/1265530.1265535))
* network analysis (e.g., Gondran and Minoux, 2008; Baras and Theodorakopoulos, 2010)
* program analysis (e.g., Cousot and Cousot, 1977; [Esparza et al., 2010](https://doi.org/10.1145/1857914.1857917); [Dolan, 2013](https://doi.org/10.1145/2500365.2500613))

Semirings enable the following separation of concerns:

1. devising a compact encoding of a set of derivations (e.g., paths in a graph, parses of a sentence), and
2. how to evaluate and aggregate the derivations (e.g., the highest-weight derivation, the total weight of all derivations).

Part 1 is specified by the algorithm (e.g., a grammar, a graph, or a dynamic program), and part 2 is given by the choice of semiring.  Modularizing by semiring gives different semantics for the same algorithm with the same time and space complexity (assuming semiring operations take unit time), and we can even reuse the implementation.<footnote>Although certain semirings may allow for additional efficiency shortcuts that others do not, certain implementations may assume properties beyond the basic semiring axioms.</footnote>

This post is background for my series on [solving weighted grammars](https://timvieira.github.io/semiring-newton/).  It is adapted from the background chapter of [my dissertation](https://timvieira.github.io/doc/2023-timv-dissertation.pdf).


## Definition

<div class="theorem" markdown="1">
**Definition.** A **semiring** $\langle \Values, \oplus, \otimes, \semizero, \semione \rangle$ is a set $\Values$ equipped with two monoid operations $\langle \Values, \oplus, \semizero \rangle$ and $\langle \Values, \otimes, \semione \rangle$ such that the following properties hold:

1. $\langle \Values, \oplus, \semizero \rangle$ is a commutative monoid
2. $\langle \Values, \otimes, \semione \rangle$ is a monoid
3. *Distributivity*: for all $\va, \vb, \vc \in \Values$, $\;\va \otimes (\vb \oplus \vc) = (\va \otimes \vb) \oplus (\va \otimes \vc)\;$ and $\;(\vb \otimes \va) \oplus (\vc \otimes \va) = (\vb \oplus \vc) \otimes \va$
4. *Multiplicative annihilation* by $\semizero$: for all $\va \in \Values$, $\;\semizero \otimes \va = \va \otimes \semizero = \semizero$
</div>

A semiring is **commutative** if $\otimes$ is commutative, and **idempotent** if $\oplus$ is idempotent ($\va \oplus \va = \va$).

The key property of a semiring is distributivity, as it allows us to factor out common subexpressions: we can safely rewrite $(\va \otimes \vb) \oplus (\va \otimes \vc)$ (three operations) into an equivalent smaller expression $\va \otimes (\vb \oplus \vc)$ (two operations).  The factored expression can be significantly smaller.

<div class="theorem" markdown="1">
**Example.** The equations below may be regarded as encoding a weighted directed graph with two states $\mathtt{x}$ and $\mathtt{y}$.  The weights in the graph have been left abstract as symbols $\va$, $\vb$, $\vc$, and $\vd$.

<figure>
<img src="/blog/images/semiring-two-state-graph.svg" alt="A weighted directed graph with states x and y: a self-loop on x with weight a, an edge from x to y with weight b, an edge from y to x with weight c, and a self-loop on y with weight d; y is the final state." style="width: 320px; max-width: 100%; display: block; margin: 0 auto;">
</figure>

$$
\begin{aligned}
\mathtt{x} &= \va \otimes \mathtt{x} \oplus \vb \otimes \mathtt{y} \\
\mathtt{y} &= \vc \otimes \mathtt{x} \oplus \vd \otimes \mathtt{y} \oplus \semione
\end{aligned}
$$

The solution to the system of equations is a map where the value associated with $\mathtt{x}$ is the set of all paths starting in state $\mathtt{x}$ and ending in state $\mathtt{y}$, and the value associated with $\mathtt{y}$ is the set of all paths starting in state $\mathtt{y}$ and ending in state $\mathtt{y}$.  In this graph, there are infinitely many paths, and yet, we can express them as a small set of equations.  Below we consider replacing the weights ($\va$, $\vb$, $\vc$, $\vd$) with values in a variety of semirings.  In each case, we describe the interpretation of the semiring values associated with $\mathtt{x}$ and $\mathtt{y}$.

* $\langle \{\dfalse, \dtrue\}, \vee, \wedge, \dfalse, \dtrue \rangle$ where $\dtrue$ is placed on each edge $\Longrightarrow$ $\mathtt{y}$ is reachable from $\mathtt{x}$, and $\mathtt{y}$ is reachable from $\mathtt{y}$.
* $\langle \mathbb{R}_{\ge 0} \cup \{\infty\}, +, \cdot, 0, 1 \rangle$ $\Longrightarrow$ the total weight of all paths from $\mathtt{x}$ to $\mathtt{y}$ and $\mathtt{y}$ to $\mathtt{y}$.
* $\langle \mathbb{R} \cup \{\infty\}, \min, +, \infty, 0 \rangle$ $\Longrightarrow$ the least-cost path from $\mathtt{x}$ to $\mathtt{y}$ and $\mathtt{y}$ to $\mathtt{y}$.
* $\langle 2^{\Sigma^*}, \cup, \cdot, \emptyset, \{\varepsilon\} \rangle$ where $\Sigma$ is the alphabet $\{a, b, c, d\}$ $\Longrightarrow$ the (regular) formal language that describes the set of paths from $\mathtt{x}$ to $\mathtt{y}$ and from $\mathtt{y}$ to $\mathtt{y}$.

This is just a small example of what we can do by swapping the semiring and keeping the equations essentially the same.
</div>


## Closed semirings and the star operation

<div class="theorem" markdown="1">
**Definition.** A **closed semiring** has an additional operator, called the (Kleene) **star** operator $\kleene{(\cdot)}$.  This operation must satisfy the following axioms for all $\va \in \Values$:

$$
\kleene{\va} = \semione \oplus \va \otimes \kleene{\va}
\quad\text{and}\quad
\kleene{\va} = \semione \oplus \kleene{\va} \otimes \va
$$
</div>

We often interpret $\kleene{\va}$ as the solution to the infinite series that arises from unrolling either of its recursive identities:

$$
\kleene{\va} = \bigoplus_{n=0}^\infty \va^n \quad\text{for } \va \in \Values
$$

The star operation is useful for solving certain linear equations.  For example, $x = \vb \oplus \va \otimes x \oplus \vc \otimes y$ can be rearranged to "solve" for $x$ *without subtraction or division*: $x = \kleene{\va} \otimes (\vb \oplus \vc \otimes y)$.  ([Part 2 of the grammar series](https://timvieira.github.io/semiring-newton/solving-linear-grammars-exactly/) uses the star of a *matrix* to solve whole systems of such equations.)

A special family of semirings that always have a star operation is those that are $K$-closed.

<div class="theorem" markdown="1">
**Definition.** For $K \in \mathbb{N}$, a semiring is **$K$-closed** if and only if<footnote>It is possible to replace $\forall K'$ with $K' = K+1$; the conditions are equivalent ([Mohri, 2002, Lemma 4](https://cs.nyu.edu/~mohri/pub/jalc.pdf)).</footnote>

$$
\forall\, \va \in \Values, \forall\, K' > K\colon \quad \bigoplus_{k=0}^{K'} \va^k = \bigoplus_{k=0}^{K} \va^k.
$$
</div>

Intuitively, $K$-closed semirings imply that we only have to sum contributions of a cycle by iterating around the cycle at most $K$ times; further iterations do not affect the value of the summation.  When $K = 0$, the condition simplifies to $\forall\, \va \in \Values\colon \semione \oplus \va = \semione$.  Examples of $0$-closed semirings include the boolean, min-plus, and Viterbi semirings.  Examples of general $K$-closed semirings include the $(K\!+\!1)$-best semirings ([Mohri, 2002, §6](https://cs.nyu.edu/~mohri/pub/jalc.pdf)) and the $K$-collapsed natural numbers.  Non-examples include the arithmetic semirings.  Each $K$-closed semiring has a simple expression for its star operation because it is no longer an infinite sum:

$$
\kleene{\va} = \bigoplus_{k=0}^K \va^k
\qquad\text{e.g., 0-closed: } \kleene{\va} = \semione, \quad \text{1-closed: } \kleene{\va} = \semione \oplus \va.
$$


## A semiring compendium

Below we list several commonly used semirings and some applications where they appear, along with their star operations.  We do not have a particular organizing principle for this list, but we hope that it may serve as a useful reference and that it illustrates that semirings can do some surprising things.

<details class="derivation" markdown="1">
<summary>Logic</summary>

* **Boolean semiring** $\langle \{\dfalse, \dtrue\}, \vee, \wedge, \dfalse, \dtrue \rangle$ is used to test reachability in graphs and, more generally, provability.  Its star operation is $\kleene{\dfalse} = \kleene{\dtrue} = \dtrue$.
* **Three-valued logic** $\langle \{\dtrue, \dfalse, \dunk\}, \max, \min, \dfalse, \dtrue \rangle$ where $\dfalse < \dunk < \dtrue$.<footnote>Often $\langle \dfalse, \dunk, \dtrue \rangle$ are encoded as $\langle -1, 0, +1 \rangle$.</footnote>
* **Fuzzy sets** $\langle [0, 1], \max, \min, 0, 1 \rangle$ is related to fuzzy sets (Zadeh, 1965).<footnote>A fuzzy set extends an ordinary set to support graded membership that indicates the extent to which a given element belongs to a given set ($0$ meaning definitely not in the set, and $1$ meaning definitely in the set).</footnote>
* **Łukasiewicz semiring** $\langle [0,1], \max, \max(0, x + y - 1), 0, 1 \rangle$ (Łukasiewicz, 1920).

These semirings are each $0$-closed; thus, their star operations are $\kleene{\va} = \semione$.
</details>

<details class="derivation" markdown="1">
<summary>Arithmetic semirings</summary>

* **Counting semiring** $\langle \mathbb{N}, +, \cdot, 0, 1 \rangle$.  As the name suggests, this semiring is used to count things, such as the number of paths between two nodes in a graph.  If $\mathbb{N}$ is augmented with $\infty$, then $\kleene{\va} = 1$ if $\va = 0$ and $\infty$ otherwise.
* **Real semiring** $\langle \mathbb{R}, +, \cdot, 0, 1 \rangle$.  If we extend $\mathbb{R}$ with $\infty$, we have $\kleene{\va} = \frac{1}{1 - \va}$ if $|\va| < 1$, $\infty$ if $\va \ge 1$, and it is undefined if $\va \le -1$.  So, unfortunately, this semiring does not have complete support for solving the infinite series.
* **Nonnegative-real semiring** $\langle \mathbb{R}_{\ge 0} \cup \{\infty\}, +, \cdot, 0, 1 \rangle$.  This semiring is used for computing the normalization constant of a probabilistic model ([Goodman, 1999](https://aclanthology.org/J99-4004/)).  Its star operation is $\kleene{\va} = \frac{1}{1 - \va}$ for $\va < 1$ and $\infty$ otherwise.
* **Log-nonnegative-real semiring** $\langle \mathbb{R} \cup \{-\infty, \infty\}, \oplus_{\log}, +, -\infty, 0 \rangle$ where $x \oplus_{\log} y \defeq \log(\exp(x) + \exp(y))$.<footnote>For speed and numerical precision, the log-space addition operation should not be computed as written; use a library function such as `numpy.logaddexp`.</footnote>  This semiring is isomorphic to the nonnegative-real semiring; however, it is often preferred as it can avoid certain numerical issues.<footnote>See also the [log-real semiring](/blog/log-real-number-class/), which extends it with a sign bit so that negative numbers can be represented.</footnote>  Its star operation is $\kleene{\va} = -\log(1 - \exp(\va))$ if $\va < 0$ and $\infty$ otherwise.
* **First-order expectation semirings** ([Eisner, 2002](https://aclanthology.org/P02-1001/)) can be used to compute first-order statistics, such as the expected error rate, the entropy, or the expected feature counts.
* **Second-order expectation semirings** ([Li and Eisner, 2009](https://aclanthology.org/D09-1005/)) can be used to compute second-order statistics, such as the variance of the error rate, the gradient of entropy, and the covariance matrix of feature counts.
* **$K$-collapsed natural numbers** $\langle \{0, \ldots, K\}, \min(K, x + y), \min(K, x \cdot y), 0, 1 \rangle$ can be used as guard rails that prevent infinite running times due to counting off to infinity.  It is isomorphic to the boolean semiring when $K = 1$.  Its star operation is $\kleene{\va} = 1$ if $\va = 0$ and $K$ otherwise.
</details>

<details class="derivation" markdown="1">
<summary>Optimization</summary>

These semirings are often used when computing the minimum or maximum value over some set, for example, the weighted set of paths in a graph.

* The **tropical semiring** (aka **min-plus**) $\langle \mathbb{R} \cup \{\infty\}, \min, +, \infty, 0 \rangle$ is used to find the shortest path or cheapest derivation.  Its star operation is $\kleene{\va} = 0$ (for $\va \ge 0$).
* The **arctic semiring** (aka **max-plus**) $\langle \mathbb{R} \cup \{-\infty\}, \max, +, -\infty, 0 \rangle$ is used to find the longest path or highest-weight derivation.  Its star operation is $\kleene{\va} = 0$ if $\va \le 0$ and $\infty$ otherwise.
* The **Viterbi semiring** (aka **max-times**) $\langle [0,1], \max, \cdot, 0, 1 \rangle$ is used to find the most probable explanation in a probabilistic model.  Its star operation is $\kleene{\va} = 1$.  If we broaden the carrier set with numbers $> 1$, we have to extend the definition to $\kleene{\va} = \infty$ for $\va > 1$.
* The **min-times semiring** $\langle \mathbb{R}_{\ge 0} \cup \{\infty\}, \min, \cdot, \infty, 1 \rangle$ can be used to find the least probable derivation.
* The **maximum-capacity semiring** $\langle \mathbb{R} \cup \{-\infty, \infty\}, \max, \min, -\infty, \infty \rangle$ can be used to find the maximum-capacity path in a graph.  It is $0$-closed, so $\kleene{\va} = \semione$.

These semirings can be augmented with additional bookkeeping to return an argmin/argmax (or even the set of argmins/argmaxes) in addition to the min/max.  It is also possible to make the elements being compared tuples and to compare them under a lexicographic ordering; this is one way to keep track of back pointers and to add tie-breaking.

The **$K$-best semiring** (top-$K$ or bottom-$K$) extends the optimization semirings to provide the $K$-best alternatives, where $K \ge 0$ is a fixed constant.<footnote>Goodman (1998) gave the $K$-best semiring, which eagerly combines the top-$K$ values; [Mohri (2002)](https://cs.nyu.edu/~mohri/pub/jalc.pdf) independently described $K$-best semirings.  [Huang and Chiang (2005)](https://aclanthology.org/W05-1506/) generalized this approach to compute the $K$-best list lazily (and without knowledge of $K$), which can be significantly faster.</footnote>  Let $\langle \mathcal{A}, \otimes, \varepsilon \rangle$ be a monoid that is partially ordered under $\preccurlyeq$, with $\otimes$ monotonic.  Each element of $\Values$ is a sorted list of at most $K$ elements of $\mathcal{A}$, and

* $A \oplus B \defeq \mathrm{sort}^K(A \cup B)$
* $A \otimes B \defeq \mathrm{sort}^K([\, a \otimes b \mid a \in A,\, b \in B \,])$
* $\semizero \defeq [\,]$ and $\semione \defeq [\varepsilon]$

where $\mathrm{sort}^K$ sorts under $\preccurlyeq$ and truncates to the $K$ smallest elements.  This semiring is $(K\!-\!1)$-closed, so we can use the truncated sum $\kleene{\va} = \bigoplus_{k=0}^{K-1} \va^k$.
</details>

<details class="derivation" markdown="1">
<summary>Formal languages and weighted formal languages</summary>

**Formal languages** $\langle 2^{\Sigma^*}, \cup, \cdot, \emptyset, \{\varepsilon\} \rangle$, where $\Sigma$ is a set of symbols, $\Sigma^*$ is the set of strings over $\Sigma$, $\varepsilon$ is the empty string, and $\cdot$ is (setwise) string concatenation.  The semiring of formal languages can be used to enumerate the set of strings accepted by a context-free grammar, a finite-state automaton, and many other formalisms.  There are, unfortunately, no shortcuts for computing its star operation.  Finite-state automata can compactly represent certain infinite sets of strings; they are closed under the union, concatenation, and star operations.<footnote>Although context-free languages are also closed under the semiring operations, their representations (context-free grammars and pushdown automata) do not have complete algorithms to test equality, severely limiting their utility as a computational representation.</footnote>

**Weighted formal languages**: The semiring of formal languages can be extended to semiring-weighted languages, where a semiring weight is assigned to each string.  Let $\langle \Values, \oplus, \otimes, \semizero, \semione \rangle$ be a base semiring and $\Sigma$ an alphabet.  The semiring of weighted formal languages $\Values\langle\!\langle \Sigma^* \rangle\!\rangle$ has as elements all functions $L \colon \Sigma^* \to \Values$, and for $L, L'$ and $x \in \Sigma^*$:

* $[L \oplus L'](x) \defeq L(x) \oplus L'(x)$
* $[L \otimes L'](x) \defeq \bigoplus_{x = y \cdot z} L(y) \otimes L'(z)$
* $\semizero(x) \defeq \semizero$ and $\semione(x) \defeq \semione$ if $x = \varepsilon$ and $\semizero$ otherwise

Formal languages are the special case of boolean weights.  Note that $\Values\langle\!\langle \Sigma^* \rangle\!\rangle$ is not commutative (concatenation does not commute), even when $\Values$ is.
</details>

<details class="derivation" markdown="1">
<summary>Matrices</summary>

We can "lift" any base semiring $\langle \Values, \oplus, \otimes, \semizero, \semione \rangle$ into a semiring over square matrices $\Values^{D \times D}$, where $D$ is a set ([Lehmann, 1977](https://doi.org/10.1016/0304-3975(77)90056-1)).  The operations are a straightforward generalization of the usual ones on $\mathbb{R}$.  Let $\boldsymbol{A}, \boldsymbol{B} \in \Values^{D \times D}$:

* $[\boldsymbol{A} + \boldsymbol{B}]_{ij} \defeq \boldsymbol{A}_{ij} \oplus \boldsymbol{B}_{ij}$
* $[\boldsymbol{A}\, \boldsymbol{B}]_{ik} \defeq \bigoplus_{j \in D} \boldsymbol{A}_{ij} \otimes \boldsymbol{B}_{jk}$
* $[\boldsymbol{0}]_{ij} \defeq \semizero$ and $[\boldsymbol{1}]_{ij} \defeq \semione$ if $i = j$ and $\semizero$ otherwise

The star operation $\kleene{\boldsymbol{A}}$ can be computed as follows.  If $|D| = 1$, then $\kleene{[a]} = [\kleene{a}]$.  Otherwise, we can decompose $\boldsymbol{A}$ into nontrivial blocks $\langle \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{D}, \boldsymbol{E} \rangle$ and apply the following blockwise recursion,

$$
\kleene{\begin{bmatrix} \boldsymbol{B} & \boldsymbol{C} \\ \boldsymbol{D} & \boldsymbol{E} \end{bmatrix}}
= \begin{bmatrix} \kleene{\boldsymbol{B}} + \kleene{\boldsymbol{B}}\, \boldsymbol{C}\, \kleene{\boldsymbol{\Delta}}\, \boldsymbol{D}\, \kleene{\boldsymbol{B}} & \kleene{\boldsymbol{B}}\, \boldsymbol{C}\, \kleene{\boldsymbol{\Delta}} \\ \kleene{\boldsymbol{\Delta}}\, \boldsymbol{D}\, \kleene{\boldsymbol{B}} & \kleene{\boldsymbol{\Delta}} \end{bmatrix}
$$

where $\boldsymbol{\Delta} = \boldsymbol{E} + \boldsymbol{D}\, \kleene{\boldsymbol{B}}\, \boldsymbol{C}$.  It is straightforward to verify that each recursive call to $\kleene{(\cdot)}$ is decreasing in size; thus, the recursive definition is well-founded.  Furthermore, if the blocks are chosen to be equal in size, then the runtime is $\mathcal{O}(|D|^3)$.
</details>

<details class="derivation" markdown="1">
<summary>Sets of vectors</summary>

* The **semiring of semilinear sets** is $\langle 2^{\mathbb{N}^\Sigma}, \cup, +, \emptyset, \{\boldsymbol{0}\} \rangle$ where $X + Y \defeq \{x + y\}_{x \in X, y \in Y}$.  This semiring can be used to compute the Parikh image of a formal language with alphabet $\Sigma$, i.e., the set of bags of symbols (forgetting their order).  See [Schlund (2016, Ch. 5)](https://mediatum.ub.tum.de/doc/1273784/1273784.pdf) for further discussion and an efficient implementation.
* The **Pareto frontier semiring** has as elements the Pareto-efficient sets of vectors, with $X \oplus Y \defeq \mathrm{pareto}(X \cup Y)$, $X \otimes Y \defeq \mathrm{pareto}(X + Y)$, $\semizero = \emptyset$, and $\semione = \{\boldsymbol{0}\}$.  The function $\mathrm{pareto}$ filters a set of vectors to those that are non-dominated.  The set of Pareto-efficient vectors represents the optimal trade-offs between different objectives.  On $\mathbb{R}_{\ge 0}^D$ it is $0$-closed, so $\kleene{X} = \semione$.
* The **semiring of convex sets in a vector space** has $X \oplus Y \defeq \mathrm{convexhull}(X \cup Y)$ and $X \otimes Y \defeq \mathrm{convexhull}(X + Y)$, with $\semizero = \emptyset$ and $\semione = \{\boldsymbol{0}\}$.  For a finite-dimensional vector space, we may restrict to polytopes, each represented by a finite set of vertices.  Dyer (2013) used this semiring for efficient minimum error rate training.
</details>

<details class="derivation" markdown="1">
<summary>Provenance semirings</summary>

**Provenance semirings** ([Green et al., 2007](https://doi.org/10.1145/1265530.1265535)) are used in database systems to validate or explain query results.  Similarly, "derivation semirings" (Goodman, 1998) can be used to encode the set of derivations for a given item.  Some examples:

* The **set semiring** $\langle 2^{\mathcal{A}}, \cup, \cap, \emptyset, \mathcal{A} \rangle$, where $\mathcal{A}$ is a base set, can be used to track the possibility of events in a system.
* The **why semiring** $\langle 2^{2^{\mathcal{A}}}, \cup, \{x \cup y\}_{x \in X, y \in Y}, \emptyset, \{\emptyset\} \rangle$.  The base set $\mathcal{A}$ is a set of annotations; the values computed with this semiring are the combinations of annotations required for a given tuple to appear in the answer to a query.
* The **lineage semiring** $\langle 2^{\mathcal{A}} \cup \{\bot\}, \cup, \cup, \bot, \emptyset \rangle$, where $\bot$ is a special element satisfying the properties of $\semizero$.  In a graph, the lineage semiring determines the subset of reachable arc labels.
</details>

<details class="derivation" markdown="1">
<summary>Other</summary>

* Polynomials and formal power series over semirings form a semiring in each case (Kuich, 1997; [Green et al., 2007](https://doi.org/10.1145/1265530.1265535)).
* Gondran and Minoux (2008) and Baras and Theodorakopoulos (2010) give many interesting semirings for network analysis, including semirings to enumerate graph cuts and bridges.
* Many works use semirings to perform dataflow analyses on programs (Cousot and Cousot, 1977; [Esparza et al., 2010](https://doi.org/10.1145/1857914.1857917); [Dolan, 2013](https://doi.org/10.1145/2500365.2500613)).
</details>
