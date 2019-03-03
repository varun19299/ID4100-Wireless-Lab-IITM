
# Probability and Generalization

We have introduced a sequence of steps to create a model using a dataset:

1. Select a model.
2. Select a loss function.
3. Fit the model by minimizing the loss on the dataset.

Thus far, we have introduced the constant model (1), a set of loss functions (2), and gradient descent as a general method of minimizing the loss (3). Following these steps will often generate a model that makes accurate predictions on the dataset it was trained on.

Unfortunately, a model that only performs well on its training data has little real-world utility. We care about the model's ability to **generalize**. Our model should make accurate predictions about the population, not just the training data. This problem seems challenging to answer—how might we reason about data we haven't seen yet?

Here we turn to the inferential power of statistics. We first introduce some mathematical tools: random variables, expectation, and variance. Using these tools, we draw conclusions about our model's long-term performance on data from our population, even data that we did not use to train the model!
