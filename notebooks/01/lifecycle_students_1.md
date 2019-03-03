

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

## The Students of Data 100

The data science lifecycle involves the following general steps:

1. **Question/Problem Formulation:** 
    1. What do we want to know or what problems are we trying to solve?  
    1. What are our hypotheses? 
    1. What are our metrics of success? <br/><br/>
1. **Data Acquisition and Cleaning:** 
    1. What data do we have and what data do we need?  
    1. How will we collect more data? 
    1. How do we organize the data for analysis?  <br/><br/>
1. **Exploratory Data Analysis:** 
    1. Do we already have relevant data?  
    1. What are the biases, anomalies, or other issues with the data?  
    1. How do we transform the data to enable effective analysis? <br/><br/>
1. **Prediction and Inference:** 
    1. What does the data say about the world?  
    1. Does it answer our questions or accurately solve the problem?  
    1. How robust are our conclusions? <br/><br/>
    
We now demonstrate this process applied to a dataset of student first names from a previous offering of Data 100. In this chapter, we proceed quickly in order to give the reader a general sense of a complete iteration through the lifecycle. In later chapters, we expand on each step in this process to develop a repertoire of skills and principles.

### Question Formulation

We would like to figure out if the student first names give
us additional information about the students themselves. Although this is a
vague question to ask, it is enough to get us working with our data and we can
make the question more precise as we go.

### Data Acquisition and Cleaning

Let's begin by looking at our data, the roster of student first names that we've downloaded from a previous offering of Data 100.

Don't worry if you don't understand the code for now; we introduce the libraries in more depth soon. Instead, focus on the process and the charts that we create.


```python
import pandas as pd

students = pd.read_csv('roster.csv')
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Keeley</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRYAN</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>276</th>
      <td>Ernesto</td>
      <td>Waitlist Student</td>
    </tr>
    <tr>
      <th>277</th>
      <td>Athan</td>
      <td>Waitlist Student</td>
    </tr>
    <tr>
      <th>278</th>
      <td>Michael</td>
      <td>Waitlist Student</td>
    </tr>
  </tbody>
</table>
<p>279 rows × 2 columns</p>
</div>



We can quickly see that there are some quirks in the data. For example, one of the student's names is all uppercase letters. In addition, it is not obvious what the Role column is for.

**In Data 100, we will study how to identify anomalies in data and apply corrections.** The differences in capitalization will cause our programs to think that `'BRYAN'` and `'Bryan'` are different names when they are identical for our purposes. Let's convert all names to lower case to avoid this.


```python
students['Name'] = students['Name'].str.lower()
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>keeley</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>1</th>
      <td>john</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bryan</td>
      <td>Student</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>276</th>
      <td>ernesto</td>
      <td>Waitlist Student</td>
    </tr>
    <tr>
      <th>277</th>
      <td>athan</td>
      <td>Waitlist Student</td>
    </tr>
    <tr>
      <th>278</th>
      <td>michael</td>
      <td>Waitlist Student</td>
    </tr>
  </tbody>
</table>
<p>279 rows × 2 columns</p>
</div>



Now that our data are in a more useful format, we proceed to exploratory data analysis.
