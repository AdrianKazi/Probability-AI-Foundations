# Probability AI Foundations

---
---

## Intro

While traditional Deep Learning relies on deterministic parameters optimized via gradient descent, Probabilistic AI explicitly represents uncertainty by modeling latent variables or parameters as random variables. Inference is performed by optimizing probabilistic objectives—most commonly the ELBO—using gradient-based optimization.

The goal is to infer quantities that are not directly observable, such as latent variables, future states, missing data, and different forms of uncertainty or risk present in the data or the environment.

***To be more intuitive:***

While traditional Deep Learning updates weights using the chain rule and gradient descent, Probabilistic AI updates weights through Bayesian inference, treating them as random variables rather than fixed values.

In theory, Probabilistic AI is based on computing the posterior distribution from a prior and a likelihood. In practice, however, the product of the prior and the likelihood produces highly irregular functions, which are usually impossible to solve analytically.

Because these functions are too complex to integrate directly, we introduce workarounds such as KL divergence and ELBO. Instead of computing the exact probability (the area under a complicated curve), we approximate the posterior with a simpler distribution that is easy to integrate, and optimize it to be as close as possible to the true posterior.

Finally, by the Law of Large Numbers, expectations computed under this approximated distribution can be estimated with arbitrarily high numerical accuracy. While this does not guarantee that the approximation equals the true posterior, it ensures that we measure the approximation itself very precisely, which is sufficient for practical inference.

---
---

## Theoretical Minimum

---

### Gaussian Distribution

The distribution shows the probability density associated with each value on the x-axis. Gaussian (Normal) Distribution has the following formula:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \Big( -\frac{(x - \mu)^2}{2\sigma^2} \Big)
$$

Where:

* $\mu$ - mean  
* $\sigma$ - standard deviation  
* $\sigma^2$ - variance  

$\mu$ and $\sigma$ are parameters of Gaussian Distribution. For standard Gaussian distribution $\mu = 0$ and $\sigma = 1$, which can be described as:

$$
N(0,1):\quad f(x) = \frac{1}{\sqrt{2\pi}} \exp \Big( -\frac{x^2}{2} \Big)
$$

![Probabilities of Gaussian Distribution for N(0,1)](plots/2.1.png)

---

### Sampling with Reparametrization Trick

Sampling takes a single value from a given distribution. For example, if our distribution is $N(\mu=2.3, \sigma=2.1)$, then we take a single sample $x$ from the available values that belong to this distribution.

However, direct sampling from $N(\mu, \sigma)$ breaks differentiability, which makes learning impossible. To fix that, we always sample from a simple, fixed distribution such as $N(0,1)$, and then deterministically transform that sample into a sample from the desired distribution.

If we draw
$$
\epsilon \sim N(0,1),
$$

then a sample from our target distribution can be computed as:

$$
z = \mu + \sigma \cdot \epsilon.
$$

This transformation is differentiable with respect to $\mu$ and $\sigma$, which allows gradients to flow through the sampling process.

![](plots/3.1.png)

---

### Conditional Probability

Conditional probability answers the question: “How probable is A, given that B is true?”
It is the core mechanism behind Bayesian inference.

Traditional (prior) probability is defined before we observe anything new, so it is static — it does not react to changes in the world.
But in real systems, once we learn new information, our beliefs must update. Conditional probability gives us the mathematical tool to do that.

A simple example: before an exam, you may estimate a 70% chance of passing based on how much you studied. This is your prior.
After taking the exam — but before receiving the grade — you might realize it went worse than expected, so your estimated probability of passing drops to 45%. This updated belief is the posterior.

The exam performance is the conditioning event that changes your probability.

$$ P(A\mid B) = \frac{P(A\cap B)}{P(B)} $$

Where:

* $P(A\cap B)$ - Probability of $A$ and $B$ happens together
* $P(B)$ - Probability of $B$ happens

![ ](plots/4.1.png)

---

### Bayes Rule

We need some tool to calculate $A \cap B$ in environments more complex than simple Venn diagrams.
Bayes Rule provides the likelihood identity:

$$ P(A\cap B) = P(B\mid A)\, P(A) $$

and from conditional probability:

$$ P(A\mid B) = \frac{P(B\mid A) P(A)}{P(B)}. $$

The purpose of this formula is to compute $P(A\mid B)$ when $P(B\mid A)$ is the quantity we can actually measure.
For example, if event $A$ is “a person has the flu” and event $B$ is “the person has a fever”, then:

$$
P(\text{flu} \mid \text{fever})
= \frac{P(\text{fever} \mid \text{flu}) \, P(\text{flu})}{P(\text{fever})}.
$$

Here, $P(B \mid A)$ or $P(\text{fever} \mid \text{flu})$ is the probability that a person has fever
given that they truly have the flu — this must be provided externally (e.g., medical statistics).
The whole idea of Bayes Rule is precisely to compute $P(\text{flu} \mid \text{fever})$ from $P(\text{fever} \mid \text{flu})$.


![ ](plots/5.1.png)

To sum up, Bayes Rule exists so that, having $P(B \mid A)$ and the prior $P(A)$,
we can compute the posterior $P(A \mid B)$. Without $P(B \mid A)$,
the posterior cannot be obtained.

---

### Integrals and Analytical Tractability

#### Integral

In Probabilistic Theory

$$ \int p(z)\;dz = 1 $$

Integral is just an area under the function.

#### Solvable Integral

$$ \int^{10}_5 2z\;dz = [z^2]^{10}_5 = 100 - 25 = 75 $$

![ ](plots/6.1.png)

* $2z$ is a straight line  
* integral is an area under this line  

#### Riemann Rectangle

We approximate integrals numerically using Riemann rectangles.
Instead of computing
$\int f(z),dz$,
we evaluate the function on a discrete grid and sum rectangular areas:

$$f(z_1)\Delta z + f(z_2)\Delta z + \dots$$

The step size
$\Delta z = z_{i+1} - z_i$
represents the width of each rectangle. As the grid becomes finer, the sum converges to the true value of the integral. This allows us to approximate expectations and KL divergence numerically when closed-form solutions are not available.

![ ](plots/6.2.png)

As we can see integral is different for every example. The smaller Riemann's rectangles, the smaller error of integral.

#### Solvable Integral with Bayesian Function

$$ \int p(y\;|\;z) \;p(z) \;dz$$
where:
$$ p(z) = x^2\ $$
$$ p(y|z) = 100 \cdot \exp{\frac{-(x-\mu)^2}{2 \cdot \sigma ^2}} $$

Bayesian functions $p(y|z)p(z)$ have 2 components:

$p(z)$ – prior, the probability of the latent variable $z$

$p(y|z)$ – likelihood, the probability of observing $y$ given $z$

$p(z)$ is a function of the latent variable $z$.
$p(y|z)$ is the probability of $y$ given a specific value of $z$.
Then $p(y|z)p(z)$ is simply the product of these two functions.

The integral $\int p(y|z)p(z)\;dz$ is the area under the product function.

![ ](plots/6.3.png)

#### Unsolvable Integral

Integral:

$$ \int exp(-x^2)\;dx $$

To solve integral means to find such $F(x)$ that $F'(x)=e^{-x^2} $ since an indefinite integral is defined as the inverse operation of differentiation.

Let

$$ u = x^2 $$

Now we need to calculate derrivative of $u$:

$$ u' = 2x $$

so

$$ du = 2x \;dx \rightarrow dx = \frac{du}{2x} $$

eventually we get:

$$ \int e^{-u}\frac{du}{2x} $$

At this point, the substitution does not eliminate the original variable
$x$.
The integral cannot be rewritten purely in terms of
$u$, which shows that the substitution fails and the integral does not admit a closed-form antiderivative.

To simplify: the integral is well-defined, but it does not admit a closed-form expression.
The only way to evaluate such an integral is through numerical (empirical) methods.
There is no elementary function that describes this integral.

![ ](plots/6.4.png)

From the plot, the integral of $\exp(-x^4)$ appears finite, as the function rapidly decays to values close to zero.
Although the tails are visually negligible, they still contribute a non-zero amount to the total area.
However, the reason this integral is not solvable in closed form is not the behavior of the tails, but the fact that no elementary function exists whose derivative equals $exp(-x^4)$. Therefore, the integral can be evaluated numerically but cannot be expressed analytically.

#### Unsolvable Integral with Bayesian Function

Unsolvable Integral:

$$ Z = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \frac{1}{1 + e^{-x}} dx $$

In Bayesian Integral we have prior and likelihood. Unsolvable Bayesian Integral can be assembled from prior integral which is solvable and likelihood integral which is solvable, but product integral of likelihood and piror is not solvable.

Prior:

$$  p(x) = N(x\,|\,\mu = 0,\sigma = 1) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} $$

Likelihood:

$$ p(y = 1 | x) = \sigma(x) = \frac{1}{1 + e^{-x}} $$


![ ](plots/6.5.png)

Although the product appears compact in the plotted range, it is not a closed function. Both the Gaussian prior and the sigmoid likelihood are strictly positive for all real values of
𝑥
x, so their product also remains strictly positive and has infinitely long tails. These tails continue to contribute to the integral, even far from the center.

Prod:
$$ \int \frac{e^{-x^2/2}}{1 + e^{-x}} dx $$

As we can see above we have values that are not 0's from around -40 to around 40. That's exactly why we can't solve this integral by simple formula.

The prior and the likelihood are individually integrable, but their product

$$ \int \frac{e^{-x^2/2}}{1 + e^{-x}} dx $$

does not have a known closed-form solution. There is no formula that solves this integral exactly, so it can only be evaluated using numerical (empirical) methods.

---

### KL divergence

$$ KL(q || p) = E_q \Big[log \frac{q(z)}{p(z)}\Big] $$

* KL isn't symetrical: $ KL(q||p) \neq KL(p||q) $
* KL $\geq$ 0, is 0 only if $q = p$

KL divergence is a measure of simmilarity between 2 distributions. It's crucial for us in Variational Inference which we use to calculate integrals of functions that are unsolvable from theoretical perspective (can be solved only numerically). In VI we take simmilar distribution to our original but one for which we can define it's integral.

#### Unsolvable Posterior


Let out original function $p(z)$ equals to:

$$ \int_{-\infty}^{\infty} e^{-x^2} log(1+e^z)\;dz $$

This function is not solvable.


![ ](plots/7.1.png)

#### Solvable Function

Although the true posterior $p(z)$ is not exactly Gaussian, its log-density is locally quadratic around the mode, which justifies using a Gaussian distribution as a first-order variational approximation.

$$ q(z) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{(z-\mu)^2}{2\sigma^2}
\right)
 $$

 As we can see, both distributions belong to the same family.
To quantify their similarity, we compute the Kullback–Leibler (KL) divergence:

$$ KL(q||p) = \int q(z) log\frac{q(z)}{p(z)} dz $$

![ ](plots/7.2.png)

#### Numerical

Note: Since $p(z)$ can be an arbitrary function, we must ensure that it integrates to $1$ in order to represent a valid probability distribution. A prerequisite for computing the KL divergence is that both $p(z)$ and $q(z)$ satisfy
$\int p(z),dz = 1$ and $\int q(z),dz = 1$.
The variational distribution $q(z)$ is chosen from a known distribution family and therefore already meets this requirement. As a result, only $p(z)$ needs to be normalized.

Note: As we know for each integral we need to specify width of the Riemann's traingle. We do this with $ dz = z[1] - z[0] $

#### Interpretation

KL = $0.3$ is relatively small, which means that the approximation $q(z)$ is reasonably close to the true distribution $p(z)$, although it is not exact. This indicates that $q(z)$ captures the main structure of $p(z)$, but still introduces some approximation error.

At this stage, we deliberately chose a simple vanilla Gaussian $q(z)$ as a tractable approximation of the intractable distribution $p(z)$. This choice allows us to compute integrals and expectations, at the cost of introducing bias.

The next step is to rewrite the KL divergence $KL(q||p)$ using the ELBO, which removes the intractable normalization constant of $p(z)$.

Finally, Variational Inference optimizes the parameters of $q(z)$ by maximizing the ELBO, yielding the best possible approximation to $p(z)$ within the chosen variational family.

* $KL \approx 0$ → nearly perfect match

* $0 < KL < 0.5$ → good / reasonable approximation

* $0.5 < KL < 2$ → noticeable difference

* $KL \gg 1$ → very poor approximation

* $KL = \infty$ → completely incompatible distributions

---

### ELBO

Evidence Lower Bound (ELBO) is a function that we optimize in Probabilistic AI.
In the previous section, we introduced KL divergence, which measures the difference between two probability distributions. Our goal is to minimize the KL divergence so that the approximate distribution is as close as possible to the true one.

Since the KL divergence is not directly tractable, we optimize the ELBO instead, which is mathematically equivalent to minimizing the KL divergence.

ELBO can be viewed in a similar way to a loss function in traditional Deep Learning.
The key difference is that:

* in Deep Learning, we minimize a loss function,
* in Variational Inference, we maximize the ELBO.

$$ ELBO(q) = E_{q(z)}[log p(x|z)] - KL(q(z)||p(z)) $$

We just need to understand $log p(x|z)$. Since we decided to use Gaussian distribution as variational, therefore if

$$ p(x|z) = N(x\;|\; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \text{exp}\Big( -\frac{(x-\mu)^2}{2\sigma^2}
\Big)$$

then

$$ log\;p(x|z) = logN(x\;|\;\mu,\sigma^2) = -\frac{(x-\mu)^2}{2\sigma^2} - log(\sqrt{2\pi}\sigma) $$

Here $z$ are our values from $p(z)$ and $q(z)$, i.e. latent variables but $x$ represents data we observe.


![ ](plots/8.1.png)

The value
$ELBO \approx -412$
is very low, which means that the current variational distribution $q(z)$ explains the observed data $x$ poorly.

This happens because the expected log-likelihood term
$\mathbb{E}_{q(z)}[\log p(x \mid z)]$
is strongly negative for many data points, and the KL penalty further decreases the total value.

A negative ELBO is normal. What matters is not its absolute value, but whether ELBO increases during optimization, which indicates that $q(z)$ is becoming a better approximation of the true posterior.

This ELBO value should be treated as a single point on the ELBO objective function. Variational Inference consists of iteratively optimizing this objective by adjusting the parameters of $q(z)$ so that the ELBO increases and approaches its maximum.

---

### Variational Inference – Approximate Posterior

We use Variational Inference to iteratively adjust the parameters of the variational distribution $q(z)$ in order to maximize the ELBO with respect to the model defined by $p(z)$.

![ ](plots/9.1.png)

ELBO goes up very fast and then flattens
This means the optimization is working and quickly finds a good explanation of the data. After that, there is not much more to improve, so learning stabilizes.

The mean of q(z) moves quickly and then stops
The model rapidly finds where the latent variable should be centered to best explain the data. Once it finds that spot, there is no reason to move further.

The variance of q(z) keeps shrinking
This shows increasing confidence. The model is saying: “I am more and more certain that the latent variable has a very specific value.” This is expected when the data is informative.

Final plot: q(z) is much sharper than p(z)
This means the posterior belief is much more precise than the prior belief. The data strongly constrains the latent variable, so uncertainty collapses.

---

### PyTorch Implementation

![ ](plots/10.1.png)

#### Interpretation

The ELBO increases smoothly and stabilizes, which shows that the optimization is working correctly and consistently improving the variational approximation. Unlike the earlier hand-crafted VI, there are no unstable jumps or plateaus.

The mean of

q(z) moves gradually toward the region supported by the data and then converges. This indicates that the model is systematically refining its belief about the latent variable rather than oscillating or collapsing.

The variance of q(z) stabilizes instead of shrinking to zero. This is important: the model retains uncertainty instead of becoming overconfident. In probabilistic inference, preserving uncertainty is a sign of a well-behaved posterior.

The final q(z) is narrower than the prior
p(z) but still aligned with it. This is exactly what we expect: the data sharpens the belief while remaining consistent with the prior structure.

**Why this works better than the manual VI implementation?**

The PyTorch implementation uses automatic differentiation instead of numerical gradients, which removes noise and instability from the optimization process.

It relies on the reparameterization trick and sampling rather than grid-based integration, avoiding discretization bias and allowing gradients to flow correctly.

Adaptive optimization (Adam) provides stable and efficient updates, whereas fixed step sizes in manual VI often lead to divergence or premature collapse.

Overall, this implementation reflects how Variational Inference is performed in real probabilistic models, while the earlier version served mainly as a conceptual and numerical demonstration.
