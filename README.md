# Probability AI Foundations

## Intro

The Probabilistic AI paradigm deals with modeling uncertainty and hidden states of the world through learning models that represent data as probability distributions rather than single values.

Its aim is to infer what we cannot observe directly—latent variables, future states, missing information, and various forms of risk and uncertainty in the data or the environment.

***To be more intuitive:***
* While traditional Deep Learning updates weights using the chain rule and gradient descent, Probabilistic AI updates weights through Bayesian inference, treating them as random variables rather than fixed values.

* In theory, Probabilistic AI is based on computing the posterior distribution from a prior and a likelihood. In practice, however, the product of the prior and likelihood produces highly irregular functions, which are usually impossible to solve analytically.

* Because these functions are too complex to integrate directly, we introduce workarounds such as KL divergence and ELBO. Instead of computing the exact probability (the area under a complicated curve), we approximate the posterior with a simpler distribution that is easy to integrate, and optimize it to be as close as possible to the true posterior.

* Finally, by the Law of Large Numbers, expectations computed under this approximated distribution can be estimated with arbitrarily high numerical accuracy. While this does not guarantee that the approximation equals the true posterior, it ensures that we measure the approximation itself very precisely, which is sufficient for practical inference.

## Theoretical Minimum

### Gaussian Distribution

The distribution shows the probability density associated with each value on the x-axis. Gaussian (Normal) Distribution has following formula:

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

### Sampling with Reparametrization Trick

Sampling takes single value from given distribution. For example if our distribution is $N(\mu=2.3, \sigma=2.1)$ then we take sample single $x$ from available $y$ values that belong to this distribution.

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

### Conditional Probability

Conditional probability answers the question: “how probable is A, given that B is true?”
It is the core mechanism behind Bayesian inference.

Traditional (prior) probability is defined before we observe anything new, so it is static — it does not react to changes in the world.
But in real systems, once we learn new information, our beliefs must update. Conditional probability gives us the mathematical tool to do that.

A simple example: before an exam, you may estimate a 70% chance of passing based on how much you studied. This is your prior.
After taking the exam — but before receiving the grade — you might realize it went worse than expected, so your estimated probability of passing drops to 45%. This updated belief is the posterior.

The exam performance is the conditioning event that changes your probability.

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)}
$$

Where:

* $P(A\cap B)$ - Probability of $A$ and $B$ happens together  
* $P(B)$ - Probability of $B$ happens

![ ](plots/4.1.png)


### Bayes Rule

We need some tool to calculate $A \cap B$ in environments more complex than simple Venn diagrams.
Bayes Rule provides the likelihood identity:

$$
P(A\cap B) = P(B\mid A)\, P(A)
$$

and from conditional probability:

$$
P(A\mid B) = \frac{P(B\mid A) P(A)}{P(B)}.
$$

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

### Integrals and Analytical Tractability

#### Integral

In Probabilistic Theory

$$
\int p(z)\;dz = 1
$$

Integral is just an area under the function.

#### Solvable Integral

$$
\int^{10}_5 2z\;dz = [z^2]^{10}_5 = 100 - 25 = 75
$$

![ ](plots/6.1.png)

* $2z$ is a straight line  
* integral is an area under this line  

#### Riemann Rectangle

We approximate integrals numerically using Riemann rectangles.
Instead of computing
$\int f(z)\,dz$,
we evaluate the function on a discrete grid and sum rectangular areas:

$f(z_1)\Delta z + f(z_2)\Delta z + \dots$

The step size
$\Delta z = z_{i+1} - z_i$
represents the width of each rectangle. As the grid becomes finer, the sum converges to the true value of the integral. This allows us to approximate expectations and KL divergence numerically when closed-form solutions are not available.

![ ](plots/6.2.png)

As we can see integral is different for every example. The smaller Riemann's rectangles, the smaller error of integral.

#### Solvable Integral with Bayesian Function

$$
\int p(y\mid z)\;p(z)\;dz
$$

where:

$$
p(z) = x^2
$$

$$
p(y|z) = 100 \cdot \exp\!\Big(\frac{-(x-\mu)^2}{2 \cdot \sigma^2}\Big)
$$

Bayesian functions $p(y|z)p(z)$ have 2 components:

* $p(z)$ – prior, the probability of the latent variable $z$
* $p(y|z)$ – likelihood, the probability of observing $y$ given $z$

The integral $\int p(y|z)p(z)\;dz$ is the area under the product function.

![ ](plots/6.3.png)

#### Unsolvable Integral

$$
\int \exp(-x^2)\;dx
$$

The integral is well-defined, but it does not admit a closed-form expression.
The only way to evaluate such an integral is through numerical (empirical) methods.
There is no elementary function that describes this integral.

![ ](plots/6.4.png)

#### Unsolvable Integral with Bayesian Function

$$
Z = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \frac{1}{1 + e^{-x}} dx
$$

Prior:

$$
p(x) = N(x\mid 0,1)
$$

Likelihood:

$$
p(y=1\mid x) = \sigma(x)
$$

![ ](plots/6.5.png)

The prior and likelihood are individually integrable, but their product does not have a known closed-form solution.

### KL divergence

$$
KL(q || p) = \mathbb{E}_q \Big[\log \frac{q(z)}{p(z)}\Big]
$$

* KL isn't symmetrical: $KL(q||p) \neq KL(p||q)$  
* KL $\geq 0$, equals $0$ only if $q = p$

KL divergence is a measure of similarity between two distributions. It's crucial for Variational Inference.

#### Unsolvable Posterior

$$
\int_{-\infty}^{\infty} e^{-x^2} \log(1+e^z)\;dz
$$

This function is not solvable.

![ ](plots/7.1.png)

#### Solvable Function

We approximate the posterior with a Gaussian:

$$
q(z) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{(z-\mu)^2}{2\sigma^2}
\right)
$$

![ ](plots/7.2.png)

#### Numerical KL

To compute KL numerically, $p(z)$ must be normalized.
Only $q(z)$ is guaranteed to integrate to 1 by construction.

#### Interpretation

* $KL \approx 0$ → nearly perfect match  
* $0 < KL < 0.5$ → good / reasonable approximation  
* $0.5 < KL < 2$ → noticeable difference  
* $KL \gg 1$ → very poor approximation  
* $KL = \infty$ → incompatible distributions  

### ELBO

Evidence Lower Bound (ELBO) is optimized instead of KL directly.

$$
ELBO(q) = \mathbb{E}_{q(z)}[\log p(x|z)] - KL(q(z)||p(z))
$$

Negative ELBO is normal. What matters is that ELBO increases during optimization.

![ ](plots/8.1.png)

### Variational Inference – Approximate Posterior

We iteratively adjust parameters of $q(z)$ to maximize ELBO.

![ ](plots/9.1.png)

**Observed behavior:**

* ELBO increases and stabilizes  
* Mean of $q(z)$ converges  
* Variance shrinks (increasing confidence)  
* Posterior is sharper than prior  

### PyTorch Implementation

![ ](plots/10.1.png)

Variational Inference implemented with:
* reparameterization trick
* Monte Carlo estimation of ELBO
* gradient-based optimization

The learned $q(z)$ converges toward a sharp posterior that explains the data significantly better than the prior.
