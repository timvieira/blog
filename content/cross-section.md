Title: Visualizing high-dimensional functions with cross-sections
date: 2014-02-12
comments: true
tags: math visualization

Last September, I gave a talk which included a bunch of two-dimensional plots of
a high-dimensional objective I was developing specialized algorithms for
optimizing. A month later, at least three of my colleagues told me that my plots
had inspired them to make similar plots. The plotting trick is really simple and
not original, but nonetheless I'll still write it up for all to enjoy.

**Example plot**: This image shows cross-sections of two related functions: a
non-smooth (black) and a smooth approximating function (blue). The plot shows
that the approximation is faithful to the overall shape, but sometimes
over-smooths. In this case, we miss the maximum, which happens near the middle
of the figure.

![Alt text](/blog/images/cross-section.png)


**Details**: Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ be a high-dimensional
function ($d \gg 2$), which you'd like to visualize. Unfortunately, you are like
me and can't see in high-dimensions what do you do?

One simple thing to do is take a nonzero vector $\boldsymbol{d} \in
\mathbb{R}^d$, take a point of interest $\boldsymbol{x}$, and build a local
picture of $f$ by evaluating it at various intervals along the chosen direction
as follows,

$f_i = f(\boldsymbol{x} + \alpha_i \ \boldsymbol{d}) \ \ $ for $\alpha_i \in [\alpha_\min, \alpha_\max]$

Of course, you'll have to pick a reasonable range and discretize it. Note,
$\boldsymbol{x}$ and $\boldsymbol{d}$ are fixed for all $\alpha_i$. Now, you can
plot $(\alpha_i,f_i)$.

**Picking directions**: There are many alternatives for picking
$\boldsymbol{d}$, my favorites are:

 1. Gradient (if it exists), this direction is guaranteed to show a local
    increase/decrease in the objective, unless it's zero.

 2. Coordinate vectors. Varying one dimension per plot.

 3. Random. I recommend directions drawn from a spherical
    Gaussian.[^sphericalgaussian] The reason being that such a vector is
    uniformly distributed across all unit-length directions (i.e., the angle of
    the vector, not it's length). We will vary the length ourselves via
    $\alpha$. It's probably best that our plots don't randomly vary in scale.

**Extension to 3d**: It's pretty easy to extend this generating 3d plots by
using 2 vectors, $\boldsymbol{d_1}$ and $\boldsymbol{d_2}$, and varying two
parameters $\alpha$ and $\beta$,

$$
f(\boldsymbol{x} + \alpha \ \boldsymbol{d_1} + \beta \ \boldsymbol{d_2})
$$

**Closing remarks**: These types of plots are probably best used to: empirically
verify/explore properties of an objective function, compare approximations, test
sensitivity to certain parameters/hyperparameters, visually debug optimization
algorithms.


Notes
-----

[^sphericalgaussian]: More formally, vectors drawn from a spherical Gaussian are
points uniformly distributed on the surface of a $d$-dimensional unit sphere,
$\mathbb{S}^d$. Sampling a vector from a spherical Gaussian is straightforward:
sample $\boldsymbol{d'} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$,
$\boldsymbol{d} = \boldsymbol{d'} / \| \boldsymbol{d'} \|_2$
