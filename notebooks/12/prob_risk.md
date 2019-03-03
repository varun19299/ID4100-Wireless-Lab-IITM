
## Risk

In a modeling scenario presented in a previous chapter, a waiter collected a dataset of tips for a particular month of work. We selected a constant model and minimized the mean squared error (MSE) loss function on this dataset, guaranteeing that our constant model outperforms all other constant models on this dataset and loss function. The constant model has a single parameter, $ \theta $. We found that the optimizing parameter $ \hat{\theta} = \text{mean}(\textbf y) $ for the MSE loss.

Although such a model makes relatively accurate predictions on its training data, we would like to know whether the model will perform well on new data from the population. To represent this notion, we introduce statistical **risk**, also known as the **expected loss**.

### Definition

A model's risk is the expected value of the model's loss on randomly chosen points from the population.

In this scenario, the population consists of all tip percentages our waiter receives during his employment, including future tips. We use the random variable $ X $ to represent a randomly chosen tip percent from the population, and the usual variable $ \theta $ to represent the constant model's prediction. Using this notation, the risk $ R(\theta) $ of our model is:

$$
\begin{aligned}
R(\theta) = \mathbb{E}\left[(X - \theta)^2\right]
\end{aligned}
$$

In the expression above, we use the MSE loss which gives the inner $ (X - \theta)^2 $ in the expectation. The risk is a function of $ \theta $ since we can change $ \theta $ as we please.

Unlike loss alone, using risk allows us to reason about the model's accuracy on the population in general. If our model achieves a low risk, our model will make accurate predictions on points from the population in the long term. On the other hand, if our model has a high risk it will in general perform poorly on data from the population.

Naturally, we would like to choose the value of $ \theta $ that makes the model's risk as low as possible. We use the variable $ \theta^* $ to represent the risk-minimizing value of $ \theta $, or the optimal model parameter for the population. To clarify, $ \theta^* $ represents the model parameter that minimizes risk while $ \hat{\theta} $ represents the parameter that minimizes dataset-specific loss.

### Minimizing the Risk

Let's find the value of $ \theta $ that minimizes the risk. Previously, we used calculus to perform this minimization. This time, we will use a mathematical trick that produces a meaningful final expression. We replace $X - \theta$ with $X - \mathbb{E}[X] + \mathbb{E}[X] - \theta$ and expand:

$$
\begin{aligned}
R(\theta) &=  \mathbb{E}[(X - \theta)^2] \\
&= \mathbb{E}\left[
  (X - \mathbb{E}[X] + \mathbb{E}[X] - \theta)^2
\right] \\
&= \mathbb{E}\left[
  \bigl( (X - \mathbb{E}[X]) + (\mathbb{E}[X] - \theta) \bigr)^2
\right] \\
&= \mathbb{E}\left[
  (X - \mathbb{E}[X])^2 + 2(X - \mathbb{E}[X])(\mathbb{E}[X] - \theta) + (\mathbb{E}[X]- \theta)^2
\right] \\
\end{aligned}
$$

Now, we apply the linearity of expectation and simplify. We use the identity $ \mathbb{E}\left[ (X - \mathbb{E}[X]) \right] = 0 $ which is roughly equivalent to stating that $ \mathbb{E}[X] $ lies at the center of the distribution of $ X $.

$$
\begin{aligned}
R(\theta) &=
  \mathbb{E}\left[ (X - \mathbb{E}[X])^2 \right]
  + \mathbb{E}\left[ 2(X - \mathbb{E}[X])(\mathbb{E}[X] - \theta) \right]
  + \mathbb{E}\left[ (\mathbb{E}[X]- \theta)^2 \right] \\
&=
  \mathbb{E}\left[ (X - \mathbb{E}[X])^2 \right]
  + 2 (\mathbb{E}[X] - \theta) \underbrace{ \mathbb{E}\left[ (X - \mathbb{E}[X]) \right]}_{= 0}
  + (\mathbb{E}[X]- \theta)^2 \\
&=
  \mathbb{E}\left[ (X - \mathbb{E}[X])^2 \right]
  + 0
  + (\mathbb{E}[X]- \theta)^2 \\
R(\theta) &=
  \mathbb{E}\left[ (X - \mathbb{E}[X])^2 \right]
  + (\mathbb{E}[X]- \theta)^2 \\
\end{aligned}
$$

Notice that the first term in the expression above is the **variance** of $ X $, $ Var(X) $, which has no dependence on $ \theta $. The second term gives a measure of how close $ \theta $ is to $ \mathbb{E}[X] $. Because of this, the second term is called the **bias** of our model. In other words, the model's risk is the bias of the model plus the variance of the quantity we are trying to predict:

$$
\begin{aligned}
R(\theta) &=
  \underbrace{(\mathbb{E}[X]- \theta)^2}_\text{bias}
  + \underbrace{Var(X)}_\text{variance}
\end{aligned}
$$

Thus, the risk is minimized when our model has no bias: $ \theta^* =  \mathbb{E}[X] $ .

#### Analysis of Risk

Notice that when our model has no bias, the risk is usually a positive quantity. This implies that even an optimal model will have prediction error. Intuitively, this occurs because a constant model will only predict a single number while $ X $ may take on any value from the population. The variance term captures the magnitude of the error. A low variance means that $ X $ will likely take a value close to $ \theta $, whereas a high variance means that $ X $ is more likely to take on a value far from $ \theta $.

### Empirical Risk Minimization

From the above analysis, we would like to set $ \theta = \mathbb{E}[X] $. Unfortunately, calculating $ \mathbb{E}[X] $ requires complete knowledge of the population. To understand why, examine the expression for $ \mathbb{E}[X] $:

$$
\begin{aligned}
\mathbb{E}[X] = \sum_{x \in \mathbb{X}} x \cdot P(X = x)
\end{aligned}
$$

$ P(X = x) $ represents the probability that $ X $ takes on a specific value from the population. To calculate this probability, however, we need to know all possible values of $ X $ and how often they appear in the population. In other words, to perfectly minimize a model's risk on a population, we need access to the population.

We can tackle this issue by remembering that the distribution of values in a large random sample will be close to the distribution of values in the population. If this is true about our sample, we can treat the sample as though it were the population itself.

Suppose we draw points at random from the sample instead of the population. Since there are $ n $ total points in the sample $ \mathbf{x} = \{ x_1, x_2, \ldots, x_n \} $, each point $ x_i $ has probability $ \frac{1}{n} $ of appearing. Now we can create an approximation for $ \mathbb{E}[X] $:

$$
\begin{aligned}
\mathbb{E}[X]
&\approx \frac{1}{n} \sum_{i=1}^n x_i = \text{mean}({\mathbf{x}})
\end{aligned}
$$

Thus, our best estimate of $ \theta^* $ using the information captured in a random sample is $ \hat{\theta} = \text{mean}(\mathbf{x}) $. We say that $ \hat{\theta} $ minimizes the **empirical risk**, the risk calculated using the sample as a stand-in for the population.

#### The Importance of Random Sampling

It is essential to note the importance of random sampling in the approximation above. If our sample is non-random, we cannot make the above assumption that the sample's distribution is similar to the population's. Using a non-random sample to estimate $ \theta^* $ will often result in a biased estimation and a higher risk.

#### Connection to Loss Minimization

Recall that we have previously shown $ \hat{\theta} = \text{mean}(\mathbf{x}) $ minimizes the MSE loss on a dataset. Now, we have taken a meaningful step forward. If our training data are a random sample, $ \hat{\theta} = \text{mean}(\mathbf{x}) $ not only produces the best model for its training data but also produces the best model for the population given the information we have in our sample.

## Summary

Using the mathematical tools developed in this chapter, we have developed an understanding of our model's performance on the population. A model makes accurate predictions if it minimizes **statistical risk**. We found that the globally optimal model parameter is:

$$
\begin{aligned}
\theta^* =  \mathbb{E}[X]
\end{aligned}
$$

Since we cannot readily compute this, we found the model parameter that minimizes the **empirical risk**.

$$
\begin{aligned}
\hat \theta = \text{mean}(\mathbf x)
\end{aligned}
$$

If the training data are randomly sampled from the population, it is likely that $ \hat{\theta} \approx \theta^* $. Thus, a constant model trained on a large random sample from the population will likely perform well on the population as well.
