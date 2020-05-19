---
layout: single
title:  "Poisson regression in python"
date:   2020-05-19
mathjax: true
---
You tried to model count data using linear regression and it felt wrong. Poisson regression is used to model count data, that is, data that can equal $$0, 1, 2, ... $$. Ordinary linear regression is not ideal in these scenarios. Linear regression usually assumes that the data can be described with a normal distribution, but a normal distribution is not constrained to the integers, nor it needs to be positive.

## Introduction
In a Poisson model, each observation corresponds to a setting like a location or a time interval. For example, the number of patients, $$y$$, that arrive to a hospital on day $$i$$ -- we would label this observation as $$y_i$$. Poisson regression is an example of a generalised linear model, so, like in ordinary linear regression or like in logistic regression, we model the variation in $$y$$ with a linear predictors $$X$$.

\begin{align}
y_i &\sim \mathrm{Poisson}(\theta_i) \newline
\theta_i &= \exp (X_i \beta)
\end{align}

Let's see how to fit a Poisson regression using Python (if you prefer R, you came to the wrong neighborhood). Fire up your jupyter server and follow along. Start by importing the necessary libraries and the raw data.


  <div class="input_area" markdown="1">
  
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats

df = pd.read_csv("../datasets/frisks.csv")
df.head()
```
</div>

You should se a table like this:

<div markdown="0" style="text-align: right">
  <table class="simpletable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stops</th>
      <th>pop</th>
      <th>past_arrests</th>
      <th>precinct</th>
      <th>eth</th>
      <th>crime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75</td>
      <td>1720</td>
      <td>191</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>1720</td>
      <td>57</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74</td>
      <td>1720</td>
      <td>599</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>1720</td>
      <td>133</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>1368</td>
      <td>62</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
  </table>
</div>

## Police stops by ethnic group
In this example,
1. The units $$i$$ are precincts (1, 2, ..., 75) and ethnic groups (_black, hispanic_ and _white_), so $$i = 1, 2, ..., 75\times3$$.
2. The outcome $$y_i$$ is the number of police stops of members of that ethnic group in that precinct.
3. The exposure $$u_i$$ is the previous year's number of arrests of members of that ethnic group in that precinct.
4. The predictors are: a constant term, 74 precinct indicators (precinct number 1 is used as the baseline so it will not be used as predictor), and 2 ethnicity indicators (one for hispanics and one for whites, with the indicator for blacks as the baseline).


The csv that we loaded is not quite in the format we need it

  <div class="input_area" markdown="1">  
```python
X = (df
    .groupby(['eth', 'precinct'])[["stops", "past_arrests"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['eth', 'precinct'])
    .assign(intercept=1)
    .sort_values(by='stops')
    .reset_index(drop=True)
)

y = X.pop("stops")
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
