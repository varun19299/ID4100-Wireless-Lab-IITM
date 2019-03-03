
## Seaborn

| Function                                                                                                | Chapter            | Description                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`sns.lmplot(x, y, data, fit_reg=True)`](https://seaborn.pydata.org/generated/seaborn.lmplot.html)      | Data Visualization | Create a scatterplot of `x` versus `y` from DataFrame `data`, and by default overlay a least-squares regression line                                                                          |
| [`sns.distplot(a, kde=True)`](https://seaborn.pydata.org/generated/seaborn.distplot.html)               | Data Visualization | Create a histogram of `a`, and by default overlay a kernel density estimator                                                                                                                  |
| [`sns.barplot(x, y, hue=None, data, ci=95)`](https://seaborn.pydata.org/generated/seaborn.barplot.html) | Data Visualization | Create a barplot of `x` versus `y` from DataFrame `data`, optionally factoring data based on `hue`, and by default drawing a 95% confidence interval (which can be turned off with `ci=None`) |
| [`sns.countplot(x, hue=None, data)`](https://seaborn.pydata.org/generated/seaborn.countplot.html)       | Data Visualization | Create a barplot of value counts of variable `x` chosen from DataFrame `data`, optionally factored by categorical variable `hue`                                                              |
| [`sns.boxplot(x=None, y, data)`](https://seaborn.pydata.org/generated/seaborn.boxplot.html)             | Data Visualization | Create a boxplot of `y`, optionally factoring by categorical variables `x`, from the DataFrame `data`                                                                                         |
| [`sns.kdeplot(x, y=None)`](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)                   | Data Visualization | If `y=None`, create a univariate density plot of `x`; if `y` is specified, create a bivariate density plot                                                                                    |
| [`sns.jointplot(x, y, data)`](https://seaborn.pydata.org/generated/seaborn.jointplot.html)              | Data Visualization | Combine a bivariate scatterplot of `x` versus `y` from DataFrame `data`, with univariate density plots of each variable overlaid on the axes                                                  |
| [`sns.violinplot(x=None, y, data)`](https://seaborn.pydata.org/generated/seaborn.violinplot.html)       | Data Visualization | Draws a combined boxplot and kernel density estimator of variable `y`, optionally factored by categorical variable `x`, chosen from DataFrame `data`                                          |
