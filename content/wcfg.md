title: Weighted Context-Free Grammars
date: 2026-11-05
tags: grammars, semirings, parsing, formal-languages
status: draft
bibtex_key: vieira2026wcfg

<macros>
\newcommand{\semiringset}{\mathbb{W}}
\newcommand{\zero}{\mathbf{0}}
\newcommand{\one}{\mathbf{1}}
\newcommand{\grammar}{\mathfrak{G}}
\newcommand{\grammarp}{\mathfrak{G}'}
\newcommand{\nts}{\mathcal{N}}
\newcommand{\terminals}{\mathcal{V}}
\newcommand{\terminalsp}{\mathcal{V}'}
\newcommand{\start}{\mathcal{S}}
\newcommand{\startp}{\mathcal{S}'}
\newcommand{\rules}{\mathcal{R}}
\newcommand{\derivations}{\mathcal{D}}
\newcommand{\derivationsp}{\mathcal{D}'}
\newcommand{\tree}{\boldsymbol{\delta}}
\newcommand{\treelabel}[1]{\ell(#1)}
\newcommand{\treeyield}[1]{\sigma(#1)}
\newcommand{\treeweight}[1]{\omega(#1)}
\newcommand{\valpha}{\boldsymbol{\alpha}}
\newcommand{\xx}{\mathbf{x}}
\newcommand{\yy}{\mathbf{y}}
\newcommand{\zz}{\mathbf{z}}
\newcommand{\ntX}{\mathrm{X}}
\newcommand{\defeq}{\overset{\text{def}}{=}}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
\newcommand{\sem}[1]{\llbracket #1 \rrbracket}
\newcommand{\Dmap}{\mathcal{M}}
</macros>

<style>
.theorem { border-left: 3px solid #c9c2b8; padding: 0.6em 1em; margin: 1.2em 0; background: #fbf8f3; }
.theorem > :first-child { margin-top: 0; }
.theorem > :last-child { margin-bottom: 0; }
</style>

<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 1.5em; border-radius: 4px; font-size: 0.95em;"><p style="margin: 0;"><strong>Draft</strong> — This article is a work in progress. Content may change.</p></div>

This post provides the necessary background on semiring-weighted context-free grammars for my series on [solving weighted grammars](https://timvieira.github.io/semiring-newton/weighted-grammars-as-algebraic-systems/).  It is adapted from the preliminaries of [Opedal, Tsipidi, Pimentel, Cotterell, and Vieira (2023), *An Exploration of Left-Corner Transformations*](https://arxiv.org/abs/2311.16258), and it keeps that paper's notation and pictures.

Weighted context-free grammars (WCFGs) define a tractable family of weighted languages, called weighted context-free languages, that is frequently used in NLP applications (see, e.g., [Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/)).


## Semirings

<div class="theorem" markdown="1">
**Definition.** A **semiring** is a tuple $\langle \semiringset, \oplus, \otimes, \zero, \one \rangle$ where $\semiringset$ is a set and the following hold:

* $\oplus$ is an associative and commutative binary operator with an identity element $\zero \in \semiringset$
* $\otimes$ is an associative binary operator with an identity element $\one \in \semiringset$
* Distributivity: $\forall a,b,c \in \semiringset$, $(a \oplus b) \otimes c = (a \otimes c) \oplus (b \otimes c)$ and $c \otimes (a \oplus b) = (c \otimes a) \oplus (c \otimes b)$
* Annihilation: $\forall a \in \semiringset$, $a \otimes \zero = \zero \otimes a = \zero$

The semiring is commutative if $\otimes$ is commutative.
</div>

We highlight a few commutative semirings and their use cases:

* boolean $\langle \{\bot, \top\}, \lor, \land, \bot, \top \rangle$: string membership in a language,
* nonnegative real $\langle \mathbb{R}_{\geq 0} \cup \{\infty\}, +, \cdot, 0, 1 \rangle$: the total probability of a string,
* Viterbi $\langle [0,1], \max, \cdot, 0, 1 \rangle$: the weight of the most likely derivation of a string.

For further reading on semirings in NLP, we recommend [Goodman (1999)](https://aclanthology.org/J99-4004) and [Huang (2008)](https://aclanthology.org/C08-5001), as well as my [semirings](/blog/semirings-a-powerful-abstraction-for-dynamic-programming/) post.


## Weighted context-free grammars

<div class="theorem" markdown="1">
**Definition.** A **weighted context-free grammar** is a tuple $\grammar = \langle \nts, \terminals, \start, \rules \rangle$ where

* $\nts$ is a set of nonterminal symbols
* $\terminals$ is a set of terminal symbols, $\terminals \cap \nts = \emptyset$
* $\start \in \nts$ is the start symbol
* $\rules$ is a bag (multiset) of weighted production rules.  Each rule $r \in \rules$ is of the form $\ntX \xrightarrow{w} \valpha$ where $\ntX \in \nts$, $\valpha \in (\terminals \cup \nts)^*$, and $w \in \semiringset$.  We assume that $\semiringset$ is a commutative semiring.
</div>

We will use the following notational conventions:

* $\mathrm{X}, \mathrm{Y}, \mathrm{Z} \in \nts$ for nonterminals
* $a, b, c \in \terminals$ for terminals
* $\xx, \yy, \zz \in \terminals^*$ for a sequence of terminals
* $\alpha, \beta, \gamma \in (\terminals \cup \nts)$ for a terminal or nonterminal symbol
* $\valpha, \boldsymbol{\beta}, \boldsymbol{\gamma} \in (\terminals \cup \nts)^*$ for a sequence of nonterminals or terminals
* $\grammar = \langle \nts, \terminals, \start, \rules \rangle$ and $\grammarp = \langle \nts', \terminalsp, \startp, \rules' \rangle$ for WCFGs.

We write $w(r)$ to access the weight of a rule $r$.

When describing a grammar and its rules, we may use the following terms.  The **size** of a grammar is $\sum_{(\ntX \to \valpha) \in \rules} (1 + |\valpha|)$.  The **arity** of a rule $\ntX \to \valpha$ is equal to $|\valpha|$; thus, a rule is **nullary** if $|\valpha| = 0$, **unary** if $|\valpha| = 1$, and so on.  For technical reasons, it is often convenient to eliminate nullary rules from the grammar.<footnote>[Opedal et al. (2023b, §F)](https://aclanthology.org/2023.acl-long.204) provides an efficient method to remove nullary rules from semiring-weighted CFGs.</footnote>


## Derivations

A **derivation** is a rooted, $(\nts \cup \terminals)$-labeled, ordered tree where each internal node must connect to its children by a production rule.<footnote>A derivation may be built by a rule with an empty right-hand side; thus, the leaves may be elements of $\nts$.  When rendering such a derivation, childlessness is marked with $\varepsilon$.</footnote>  To access the root **label** of a derivation $\tree$, we write $\treelabel{\tree}$.  The **set of all derivations** of $\grammar$ is the smallest set $\derivations$ that contains every terminal ($\terminals \subseteq \derivations$) and is closed under the following construction: for every rule $(\ntX \to \alpha_1 \cdots \alpha_K) \in \rules$ and derivations of $\alpha_1, \ldots, \alpha_K$ in $\derivations$, the tree with root $\ntX$ and those derivations as its children is in $\derivations$.

<figure>
<img src="/blog/images/wcfg-derivation-node.svg" alt="A derivation with root X whose children are subderivations rooted at alpha_1 through alpha_K, each drawn as a triangle." style="width: 340px; max-width: 100%; display: block; margin: 0 auto;">
<figcaption style="text-align: center; font-style: italic; font-size: 0.9em; margin: 0.4em 0 1.2em;">A derivation built by the rule $\ntX \to \alpha_1 \cdots \alpha_K$ from subderivations rooted at $\alpha_1, \ldots, \alpha_K$.</figcaption>
</figure>

For example, here is a derivation of the sentence "My sister's diploma arrived" (subtrees whose details don't matter are drawn as triangles over their yields):

<figure>
<img src="/blog/images/wcfg-example-derivation.svg" alt="A derivation of the sentence My sister's diploma arrived: S over NP and VP; the NP over PossP and NN; the PossP over NP (My sister) and 's; NN over diploma; VP over arrived." style="width: 480px; max-width: 100%; display: block; margin: 0 auto;">
</figure>

Let $\tree$ be a derivation built by the rule $\ntX \to \alpha_1 \cdots \alpha_K$ from subderivations $\tree_1, \ldots, \tree_K$ (as in the figure above).  The **yield** $\treeyield{\tree} \in \terminals^*$ of a derivation is

* **if** $\tree \in \terminals$: $\treeyield{\tree} \defeq \treelabel{\tree} = a$
* **else:** $\treeyield{\tree} \defeq \treeyield{\tree_1} \circ \cdots \circ \treeyield{\tree_K}$

where $\circ$ denotes concatenation.  The **weight** $\treeweight{\tree} \in \semiringset$ of a derivation is

* **if** $\tree \in \terminals$: $\treeweight{\tree} \defeq \one$
* **else:** $\treeweight{\tree} \defeq w(\ntX \to \alpha_1 \cdots \alpha_K) \otimes \treeweight{\tree_1} \otimes \cdots \otimes \treeweight{\tree_K}$


## Weighted languages

The **weighted language** of $\alpha \in (\terminals \cup \nts)$ is a function $\grammar_\alpha \colon \terminals^* \to \semiringset$ defined as follows:

$$
\grammar_\alpha(\xx) \defeq \bigoplus_{\tree \in \derivations_\alpha(\xx)} \treeweight{\tree}
$$

where $\derivations_\alpha \defeq \{ \tree \in \derivations \colon \treelabel{\tree} = \alpha \}$ denotes the subset of $\derivations$ containing trees labeled $\alpha$, and $\derivations_\alpha(\xx) \defeq \{ \tree \in \derivations_\alpha \colon \treeyield{\tree} = \xx \}$ denotes those with yield $\xx$.  In words, the value of a string $\xx \in \terminals^*$ in the weighted language $\grammar_\alpha$ is the $\oplus$-sum of the weights of all trees in $\derivations_\alpha$ with $\xx$ as their yield.  The weighted language of the *grammar* $\grammar$ is $\grammar(\xx) \defeq \grammar_\start(\xx)$.  Given a set of symbols $\mathcal{X}$, we write $\derivations_{\mathcal{X}}$ as shorthand for $\bigcup_{\alpha \in \mathcal{X}} \derivations_\alpha$.  Lastly, let $\sem{\tree}$ denote the weighted language generated by the tree $\tree$.<footnote>Formally, $\sem{\tree}(\xx) \defeq \treeweight{\tree}$ if $\xx = \treeyield{\tree}$ else $\zero$.</footnote>

We define the following operations on weighted languages $\grammar$ and $\grammarp$, for $\xx \in (\terminals \cup \terminalsp)^*$:

* Union: $[\grammar \oplus \grammarp](\xx) \defeq \grammar(\xx) \oplus \grammarp(\xx)$
* Concatenation: $[\grammar \circ \grammarp](\xx) \defeq \bigoplus_{\yy \circ \zz = \xx} \grammar(\yy) \otimes \grammarp(\zz)$

Note that these operations form a (noncommutative) semiring over weighted languages where $\zero$ is the language that assigns weight zero to all strings and $\one$ is the language that assigns one to the empty string and zero to other strings.

The weighted language of $\grammar$ may also be expressed as a certain solution<footnote>Note that the system of equations does not necessarily have a unique solution.  In the case of an $\omega$-continuous semiring, the definition of $\grammar_\alpha$ above coincides with the *smallest* solution to these equations under the natural ordering ([Droste and Kuich, 2009](https://doi.org/10.1007/978-3-642-01492-5_1)).</footnote> to the following system of equations:

$$
\begin{aligned}
\grammar_a &= \one_a & \forall a \in \terminals \\
\grammar_\ntX &= \bigoplus_{(\ntX \xrightarrow{w} \beta_1 \cdots \beta_K) \in \rules} w \circ \grammar_{\beta_1} \circ \cdots \circ \grammar_{\beta_K} & \forall \ntX \in \nts
\end{aligned}
$$

where $\one_a$ is the weighted language that assigns $\one$ to the string $a$ and $\zero$ to other strings.

These equations are where the [grammar series](https://timvieira.github.io/semiring-newton/weighted-grammars-as-algebraic-systems/) picks up.  Its algorithms solve the special case where we forget the strings and only keep the total weight, $\bigoplus_{\xx} \grammar_\ntX(\xx)$.  In that case, the equations live in the (commutative) semiring $\semiringset$ itself.

<details class="derivation" markdown="1">
<summary>Useless symbols, trimming, and equivalence of grammars</summary>

We say that $\alpha$ is **useless** if there does not exist a derivation $\tree \in \derivations_\start$ that has a subderivation $\tree'$ with $\treelabel{\tree'} = \alpha$.  We define **trimming** $\mathrm{TRIM}(\grammar)$ as removing each useless nonterminal and any rule in which they participate.  It is easy to see that trimming does not change the weighted language of the grammar because no useless nonterminals participate in a derivation rooted at $\start$.  We can trim useless rules in linear time using well-known algorithms (Hopcroft and Ullman, 1979).

We say that grammars $\grammar$ and $\grammarp$ are **equal** ($\grammar = \grammarp$) if they have the same tuple representation after trimming.  We say they are **equivalent** ($\grammar \equiv \grammarp$) if they define the same weighted language.<footnote>I.e., $(\grammar \equiv \grammarp) \Longleftrightarrow \forall \xx \in (\terminals \cup \terminalsp)^* \colon \grammar_\start(\xx) = \grammarp_{\startp}(\xx)$.</footnote>  We say that they are **$\mathcal{X}$-bijectively equivalent** ($\grammar \equiv_{\mathcal{X}} \grammarp$) if a structure-preserving bijection of type $\Dmap \colon \derivations_{\mathcal{X}} \to \derivationsp_{\mathcal{X}}$ exists.  The mapping $\Dmap$ is **structure-preserving** if ($\forall \tree \in \derivations_{\mathcal{X}}$) it is (i) label-preserving ($\treelabel{\tree} = \treelabel{\Dmap(\tree)}$), (ii) yield-preserving ($\treeyield{\tree} = \treeyield{\Dmap(\tree)}$), and (iii) weight-preserving ($\treeweight{\tree} = \treeweight{\Dmap(\tree)}$).  Suppose $\grammar \equiv_{\mathcal{X}} \grammarp$, $\start \in \mathcal{X}$, and $\start = \startp$; then $\grammar \equiv \grammarp$, but not conversely.  A benefit of this stronger notion of equivalence is that derivations in $\grammar$ and $\grammarp$ are interconvertible: we can parse in $\grammar$ and convert to a parse in $\grammarp$ and vice versa.
</details>


## Further reading: the left-corner transformation

Grammar transformations are functions that map one context-free grammar to another, such as nullary rule removal, rule binarization, and conversion to normal forms.  The left-corner transformation (Rosenkrantz and Lewis, 1970) maps the derivation trees of an original grammar into isomorphic trees in the transformed grammar.  The trees of the transformed grammar are such that the base subtree of a left-recursive chain (the left corner) is *hoisted* up in a derivation tree while replacing the left-recursive path to the left corner with a right-recursive path to an empty constituent:

<figure>
<img src="/blog/images/wcfg-left-corner-example.svg" alt="Left: the derivation of My sister's diploma arrived, with left recursion from NP down to NP. Right: the corresponding derivation after the left-corner transformation, where the lower NP (My sister) is hoisted to be the first child of S, and a right-recursive chain of S/NP, S/PossP, S/NP, S/S nonterminals leads to an empty constituent." style="width:100%">
<figcaption style="text-align: center; font-style: italic; font-size: 0.9em; margin: 0.4em 0 1.2em;"><em>Left:</em> an example derivation containing left-recursive rules.  <em>Right:</em> the corresponding derivation after the left-corner transformation.  Observe that (i) the lower NP subtree has been hoisted up the tree, and (ii) the <em>left</em> recursion from NP to NP has been replaced with <em>right</em> recursion from 𝒮/NP to 𝒮/NP.</figcaption>
</figure>

The paper ([Opedal et al., 2023](https://arxiv.org/abs/2311.16258)) generalizes the left-corner transformation to semiring-weighted grammars, relates it to the speculation transformation, and proves that the transformed grammar's derivations are in bijection with the original's.  The code is in [`rycolab/left-corner`](https://github.com/rycolab/left-corner), the WCFG library used by the grammar series.


## References

* Andreas Opedal, Eleftheria Tsipidi, Tiago Pimentel, Ryan Cotterell, and Tim Vieira.  [An Exploration of Left-Corner Transformations](https://arxiv.org/abs/2311.16258).  EMNLP 2023.
* Andreas Opedal, Ran Zmigrod, Tim Vieira, Ryan Cotterell, and Jason Eisner.  [Efficient Semiring-Weighted Earley Parsing](https://aclanthology.org/2023.acl-long.204).  ACL 2023.
* Joshua Goodman.  [Semiring Parsing](https://aclanthology.org/J99-4004).  Computational Linguistics, 1999.
* Liang Huang.  [Advanced Dynamic Programming in Semiring and Hypergraph Frameworks](https://aclanthology.org/C08-5001).  COLING 2008 tutorial notes.
* Manfred Droste and Werner Kuich.  [Semirings and Formal Power Series](https://doi.org/10.1007/978-3-642-01492-5_1).  Handbook of Weighted Automata, 2009.
* John E. Hopcroft and Jeffrey D. Ullman.  Introduction to Automata Theory, Languages and Computation.  1979.
* Daniel J. Rosenkrantz and Philip M. Lewis II.  Deterministic Left Corner Parsing.  1970.
* Dan Jurafsky and James H. Martin.  [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/).
