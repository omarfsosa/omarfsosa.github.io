---
layout: single
title:  "Generalised linear models: Part 1"
date:   2018-01-02
mathjax: true
---

Linear regression predicts $$y$$ as linear combination of the predictors, $$X$$,

\begin{align}
\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_D X_D = X\beta .
\end{align}

The most common way in which this is done, is by assuming that $$y$$ follows a normal distribution with mean given by $$X\beta$$, and standard deviation $$\sigma$$.

\begin{align}
y \sim \mathcal{N}(X\beta, \sigma),
\end{align}

A _generalised linear model_ (GLM) involves
1. A link function, $$g$$, such that $$\hat{y} = g^{-1}(X\beta)$$ that are used to model the data.
2. A data distribution $$p(y|\hat{y})$$,

Just like traditional linear models have a variance parameter, a GLM may involve other parameters which we'll mention later on. Of course, linear models are a trivial special case of GLM's where the link function is the identity, $$g(x) = x$$, and the data distribution is the normal distribution.

But chances are you've already came across a non-trivial example of a GLM: _logistic regression_. In logistic regression, the link function is the logistic function (duh...),

\begin{align}
\mathrm{logit}^{-1}(x) = \mathrm{logistic}(x) = \frac{1}{1 + e^{-x}}
\end{align}

and the data distribution is the Bernoulli distribution because the data $$y$$ is binary. So we write,

\begin{align}
y \sim \mathrm{Bernoulli}\left({\mathrm{logit}^{-1}(X\beta})\right)
\end{align}


The need for a GLM appears, for example, when we try to model data on _counts_. Count data will always be integer-valued and positive, so if we use traditional linear regression, and therefore assume that the data follows a normal distribution, we are already doing something wrong since the normal distribution can take any real value.

In this series, you'll learn about 3 different GLMs:
1. __Poisson model__. This is used to model count data, where each data point can equal $$0, 1, 2, ...$$. 
2. __Logistic-binomial model___. This is used to model data where each point counts the number of successes out of a number of trials (the number of trials need not to be the same for each observation).
3. __Robust regression model__. Here, the usual normal distribution in a traditional linear regression is replaced with a student-t distribution. This allows for smooth handling of ocassional outliers.




  <div class="input_area" markdown="1">
  
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
df = pd.read_csv("../datasets/frisks.csv")
```

  </div>
  
## Poisson regression, exposure and overdispersion


  <div class="input_area" markdown="1">
  
```python
model_no_indicators = sm.GLM(
    df2.stops,
    df2[["intercept"]],
    offset=np.log(df2.past_arrests),
    family=sm.families.Poisson(),
)
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
result_no_indicators = model_no_indicators.fit()
result_no_indicators.summary()
```

  </div>
  



  <div markdown="0">
  <table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>stops</td>      <th>  No. Observations:  </th>  <td>   225</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   224</td> 
</tr>
<tr>
  <th>Model Family:</th>         <td>Poisson</td>     <th>  Df Model:          </th>  <td>     0</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -23913.</td>
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 18 May 2020</td> <th>  Deviance:          </th> <td>  46120.</td>
</tr>
<tr>
  <th>Time:</th>                <td>21:51:33</td>     <th>  Pearson chi2:      </th> <td>4.96e+04</td>
</tr>
<tr>
  <th>No. Iterations:</th>          <td>5</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -0.5877</td> <td>    0.003</td> <td> -213.058</td> <td> 0.000</td> <td>   -0.593</td> <td>   -0.582</td>
</tr>
</table>
  </div>
  



  <div class="input_area" markdown="1">
  
```python
model_with_ethnicity = sm.GLM(
    df2.stops,
    df2[['intercept', 'eth_2', 'eth_3']],
    offset=np.log(df2.past_arrests),
    family=sm.families.Poisson(),
)
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
result_with_ethnicity = model_with_ethnicity.fit()
result_with_ethnicity.summary()
```

  </div>
  



  <div markdown="0">
  <table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>stops</td>      <th>  No. Observations:  </th>  <td>   225</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   222</td> 
</tr>
<tr>
  <th>Model Family:</th>         <td>Poisson</td>     <th>  Df Model:          </th>  <td>     2</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -23572.</td>
</tr>
<tr>
  <th>Date:</th>            <td>Mon, 18 May 2020</td> <th>  Deviance:          </th> <td>  45437.</td>
</tr>
<tr>
  <th>Time:</th>                <td>21:51:38</td>     <th>  Pearson chi2:      </th> <td>4.94e+04</td>
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -0.5881</td> <td>    0.004</td> <td> -155.396</td> <td> 0.000</td> <td>   -0.596</td> <td>   -0.581</td>
</tr>
<tr>
  <th>eth_2</th>     <td>    0.0702</td> <td>    0.006</td> <td>   11.584</td> <td> 0.000</td> <td>    0.058</td> <td>    0.082</td>
</tr>
<tr>
  <th>eth_3</th>     <td>   -0.1616</td> <td>    0.009</td> <td>  -18.881</td> <td> 0.000</td> <td>   -0.178</td> <td>   -0.145</td>
</tr>
</table>
  </div>
  



  <div class="input_area" markdown="1">
  
```python
model_with_ethnicity_and_precinct = sm.GLM(
    df2.stops,
    df2.drop(columns=["stops", "eth_1", "precinct_1"]),
    offset=np.log(df2.past_arrests),
    family=sm.families.Poisson(),
)
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
result_with_ethnicity_and_precinct = model_with_ethnicity_and_precinct.fit()
```

  </div>
  
### Overdispersion


  <div class="input_area" markdown="1">
  
```python
def group_residuals(y_true, y_pred, n_groups=20, offset=0):
    residuals = y_true - y_pred
    quantiles = pd.qcut(y_true, n_groups, labels=False)
    groups = residuals.groupby(quantiles)
    mean = groups.mean()
    std = groups.std()
    return mean.index + offset, mean.values, std.values
```

  </div>
  
The standardised residuals are already stored in the fitted model in the attribute `resid_pearson`, so we don't need to compute these by hand.
