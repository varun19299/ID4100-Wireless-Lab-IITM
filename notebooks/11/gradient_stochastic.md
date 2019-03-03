

```python
# HIDDEN
import warnings
# Ignore numpy dtype warnings. These warnings are caused by an interaction
# between numpy and Cython and can be safely ignored.
# Reference: https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```

## Stochastic Gradient Descent

In this section, we discuss a modification to gradient descent that makes it much more useful for large datasets. The modified algorithm is called **stochastic gradient descent**.

Recall gradient descent updates our model parameter $ \theta $ by using the gradient of our chosen loss function. Specifically, we used this gradient update formula:

$$
{\theta}^{(t+1)} = \theta^{(t)} - \alpha \cdot \nabla_{\theta} L(\theta^{(t)}, \textbf{y})
$$

In this equation:

- $\theta^{(t)}$ is our current estimate of $\theta^*$ at the $t$th iteration
- $\alpha$ is the learning rate
- $\nabla_{\theta} L(\theta^{(t)}, \textbf{y})$ is the gradient of the loss function
- We compute the next estimate $\theta^{(t+1)}$ by subtracting the product of $\alpha$ and $\nabla_{\theta} L(\theta, \textbf{y})$ computed at $\theta^{(t)}$

### Limitations of Batch Gradient Descent

In the expression above, we calculate $\nabla_{\theta}L(\theta, \textbf{y})$ using the average gradient of the loss function $\ell(\theta, y_i)$ using **the entire dataset**. In other words, each time we update $ \theta $ we consult all the other points in our dataset as a complete batch. For this reason, the gradient update rule above is often referred to as **batch gradient descent**. 

Unfortunately, we often work with large datasets. Although batch gradient descent will often find an optimal $ \theta $ in relatively few iterations, each iteration will take a long time to compute if the training set contains many points.

### Stochastic Gradient Descent

To circumvent the difficulty of computing a gradient across the entire training set, stochastic gradient descent approximates the overall gradient using **a single randomly chosen data point**. Since the observation is chosen randomly, we expect that using the gradient at each individual observation will eventually converge to the same parameters as batch gradient descent.

Consider once again the formula for batch gradient descent:

$$
{\theta}^{(t+1)} = \theta^{(t)} - \alpha \cdot \nabla_{\theta} L(\theta^{(t)}, \textbf{y})
$$

In this formula, we have the term $\nabla_{\theta} L(\theta^{(t)}, \textbf{y})$, the average gradient of the loss function across all points in the training set. That is:

$$
\begin{aligned}
\nabla_{\theta} L(\theta^{(t)}, \textbf{y}) &= \frac{1}{n} \sum_{i=1}^{n} \nabla_{\theta} \ell(\theta^{(t)}, y_i)
\end{aligned}
$$

Where $ \ell(\theta, y_i) $ is the loss at a single point in the training set. To conduct stochastic gradient descent, we simply replace the average gradient with the gradient at a single point. The gradient update formula for stochastic gradient descent is:

$$
{\theta}^{(t+1)} = \theta^{(t)} - \alpha \cdot \nabla_{\theta} \ell(\theta^{(t)}, y_i)
$$

In this formula, $ y_i $ is chosen randomly from $ \textbf{y} $. Note that choosing the points randomly is critical to the success of stochastic gradient descent! If the points are not chosen randomly, stochastic gradient descent may produce significantly worse results than batch gradient descent.

We most commonly run stochastic gradient descent by shuffling the data points and using each one in its shuffled order until one complete pass through the training data is completed. If the algorithm hasn't converged, we reshuffle the points and run another pass through the data. Each **iteration** of stochastic gradient descent looks at one data point; each complete pass through the data is called an **epoch**.

#### Using the MSE Loss

As an example, we derive the stochastic gradient descent update formula for the mean squared loss. Recall the definition of the mean squared loss:

$$
\begin{aligned}
L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n}(y_i - \theta)^2
\end{aligned}
$$

Taking the gradient with respect to $ \theta $, we have:

$$
\begin{aligned}
\nabla_{\theta}  L(\theta, \textbf{y})
&= \frac{1}{n} \sum_{i = 1}^{n} -2(y_i - \theta)
\end{aligned}
$$

Since the above equation gives us the average gradient loss across all points in the dataset, the gradient loss on a single point is simply the piece of the equation that is being averaged:

$$
\begin{aligned}
\nabla_{\theta}  \ell(\theta, y_i)
&= -2(y_i - \theta)
\end{aligned}
$$

Thus, the batch gradient update rule for the MSE loss is:

$$
\begin{aligned}
{\theta}^{(t+1)} = \theta^{(t)} - \alpha \cdot \left( \frac{1}{n} \sum_{i = 1}^{n} -2(y_i - \theta) \right)
\end{aligned}
$$

And the stochastic gradient update rule is:

$$
\begin{aligned}
{\theta}^{(t+1)} = \theta^{(t)} - \alpha \cdot \left( -2(y_i - \theta) \right)
\end{aligned}
$$

### Behavior of Stochastic Gradient Descent

Since stochastic descent only examines a single data point a time, it will likely update $ \theta $ less accurately than a update from batch gradient descent. However, since stochastic gradient descent computes updates much faster than batch gradient descent, stochastic gradient descent can make significant progress towards the optimal $ \theta $ by the time batch gradient descent finishes a single update.

In the image below, we show successive updates to $ \theta $ using batch gradient descent. The darkest area of the plot corresponds to the optimal value of $ \theta $ on our training data, $ \hat{\theta} $.

(This image technically shows a model that has two parameters, but it is more important to see that batch gradient descent always takes a step towards $ \hat{\theta} $.)

<img style="max-width: 500px" src="https://raw.githubusercontent.com/DS-100/textbook/master/assets/gd.png"></img>

Stochastic gradient descent, on the other hand, often takes steps away from $ \hat{\theta} $! However, since it makes updates more often, it often converges faster than batch gradient descent.

<img style="max-width: 500px" src="https://raw.githubusercontent.com/DS-100/textbook/master/assets/sgd.png"></img>

### Defining a Function for Stochastic Gradient Descent

As we previously did for batch gradient descent, we define a function that computes the stochastic gradient descent of the loss function. It will be similar to our `minimize` function but we will need to implement the random selection of one observation at each iteration.


```python
def minimize_sgd(loss_fn, grad_loss_fn, dataset, alpha=0.2):
    """
    Uses stochastic gradient descent to minimize loss_fn.
    Returns the minimizing value of theta once theta changes
    less than 0.001 between iterations.
    """
    NUM_OBS = len(dataset)
    theta = 0
    np.random.shuffle(dataset)
    while True:
        for i in range(0, NUM_OBS, 1):
            rand_obs = dataset[i]
            gradient = grad_loss_fn(theta, rand_obs)
            new_theta = theta - alpha * gradient
        
            if abs(new_theta - theta) < 0.001:
                return new_theta
        
            theta = new_theta
        np.random.shuffle(dataset)
```

### Mini-batch Gradient Descent

**Mini-batch gradient descent** strikes a balance between batch gradient descent and stochastic gradient descent by increasing the number of observations that we select at each iteration. In mini-batch gradient descent, we use a few data points for each gradient update instead of a single point.

We use the average of the gradients of their loss functions to construct an estimate of the true gradient of the cross entropy loss. If $\mathcal{B}$ is the mini-batch of data points that we randomly sample from the $n$ observations, the following approximation holds.

$$
\nabla_\theta L(\theta, \textbf{y}) \approx \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\nabla_{\theta}\ell(\theta, y_i)
$$

As with stochastic gradient descent, we perform mini-batch gradient descent by shuffling our training data and selecting mini-batches by iterating through the shuffled data. After each epoch, we re-shuffle our data and select new mini-batches.

While we have made the distinction between stochastic and mini-batch gradient descent in this textbook, stochastic gradient descent is sometimes used as an umbrella term that encompasses the selection of a mini-batch of any size. 


#### Selecting the Mini-Batch Size

Mini-batch gradient descent is most optimal when running on a Graphical Processing Unit (GPU) chip found in some computers. Since computations on these types of hardware can be executed in parallel, using a mini-batch can increase the accuracy of the gradient without increasing computation time. Depending on the memory of the GPU, the mini-batch size is often set between 10 and 100 observations.

### Defining a Function for Mini-Batch Gradient Descent

A function for mini-batch gradient descent requires the ability to select a batch size. Below is a function that implements this feature.


```python
def minimize_mini_batch(loss_fn, grad_loss_fn, dataset, minibatch_size, alpha=0.2):
    """
    Uses mini-batch gradient descent to minimize loss_fn.
    Returns the minimizing value of theta once theta changes
    less than 0.001 between iterations.
    """
    NUM_OBS = len(dataset)
    assert minibatch_size < NUM_OBS
    
    theta = 0
    np.random.shuffle(dataset)
    while True:
        for i in range(0, NUM_OBS, minibatch_size):
            mini_batch = dataset[i:i+minibatch_size]
            gradient = grad_loss_fn(theta, mini_batch)
            new_theta = theta - alpha * gradient
            
            if abs(new_theta - theta) < 0.001:
                return new_theta
            
            theta = new_theta
        np.random.shuffle(dataset)
```

## Summary

We use batch gradient descent to iteratively improve model parameters until the model achieves minimal loss. Since batch gradient descent is computationally intractable with large datasets, we often use stochastic gradient descent to fit models instead. When using a GPU, mini-batch gradient descent can converge more quickly than stochastic gradient descent for the same computational cost. For large datasets, stochastic gradient descent and mini-batch gradient descent are often preferred to batch gradient descent for their faster computation times.
