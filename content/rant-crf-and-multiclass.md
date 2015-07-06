title: Multiclass logistic regression and conditional random fields are the same thing
date: 2015-04-29
comments: true
tags: machine-learning, rant, crf

A short rant: multiclass logistic regression and conditional random fields (CRF)
are the same thing. This comes to a surprise to many people because CRFs tend to
be surrounded by additional "stuff."

Multiclass logistic regression is simple. The goal is to predict the correct
label $y^*$ from handful of labels $\mathcal{Y}$ given the observation $x$ based
on features $\phi(x,y)$. Training this model typically requires computing the
gradient:

$$
\phi(x,y^*) - \sum_{y \in \mathcal{Y}} p(y|x) \phi(x,y)
$$

where
$$
\begin{eqnarray*}
p(y|x) &=& \frac{1}{Z(x)} \exp(\theta^\top \phi(x,y)) & \ \ \ \ \text{and} \ \ \ \ &
Z(x) &=& \sum_{y \in \mathcal{Y}} \exp(\theta^\top \phi(x,y))
\end{eqnarray*}
$$

At test-time, we often take the highest-scoring label under the model.

$$
\hat{y}(x) = \textbf{argmax}_{y \in \mathcal{Y}} \theta^\top \phi(x,y)
$$

A conditional random field is exactly multiclass logistic regression. The only
difference is that the sum and argmax are inefficient to compute naively (i.e.,
by enumeration). This point is often lost when people first learn about
CRFs. Some people never make this connection.

Here's some stuff you'll see once we start talking about CRFs:

1. Inference algorithms (e.g., Viterbi decoding, forward-backward, Junction
   tree)

2. Graphical models (factor graphs, Bayes nets, Markov random fields)

3. Model templates (i.e., repeated feature functions)

In the logistic regression case, we'd never use the term "inference" to describe
the "sum" and "max" over a handful of categories. Once we move to a structured
label space, this term gets throw around. (BTW, this isn't "statistical
inference," just algorithms to compute sum and max over $\mathcal{Y}$.)

Graphical models establish a notation and structural properties which allow
efficient inference -- things like cycles and treewidth.

Model templating is the only essential trick to move from logistic regression to
a CRF. Templating "solves" the problem that not all training examples have the
same "size" -- the set of outputs $\mathcal{Y}(x)$ now depends on $x$. A model
template specifies how to compute the features for an entire output, by looking
at interactions between subsets of variables.

$$
\phi(x,\boldsymbol{y}) = \sum_{\alpha \in A(x)} \phi_\alpha(x,
\boldsymbol{y}_\alpha)
$$

where $\alpha$ is a labeled subset of variables often called a factor and
$\boldsymbol{y}_\alpha$ is the subvector containing values of variables
$\alpha$. Basically, the feature function $\phi$ gets to look at some subset of
the variables being predicted $y$ and the entire input $x$. The ability to look
at more of $y$ allows the model to make more coherent predictions.

Anywho, it's often useful to take a step back and think about what you are
trying to compute instead of how you're computing it. In this post, this allowed
us see the similarity between logistic regression and CRFs even though they seem
quite different.
