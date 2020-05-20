---
layout: single
title:  "Poisson regression in python"
date:   2020-05-19
mathjax: true
---
You tried to model count data using linear regression and it felt wrong. All your observations are integers and yet your model assumes continuous data. Noise seems to be larger when your observations take large values, but your model assumes the same amount of variance all across the board. Even worse, when your observations take small values, sometimes your model predicts negative values! So when no one else was watching you truncated your predictions at zero, `y_pred = max(0, y_pred)`. You have not sleep since then. 

Finally you realise: you need to model your data using a Poisson distribution! After doing some <s>googling</s> thorough research, you find that every single tutorial and reference out there uses R instead of Python. You're now considering installing RStudio -- but maybe not, since you have deadline ahead of you and learning a new programming language is not going to happen in one day. 

Fear not. Here you will learn how to do Poisson regression, and all within the comfort of your beloved Python.
I'll show you how to model the same example that is treated in chapter 6 of [this book](http://www.stat.columbia.edu/~gelman/arm/)[^1]. But, yes, we'll do in Python. So fire up a jupyter notebook and follow along.

[^1]: _Data Analysis Using Regression and Multilevel/Hierarchical Models_ by Andrew Gelman and Jennifer Hill.

## Introduction
Start by importing the necessary libraries and the raw data.

 <div class="input_area" markdown="1">
  
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats

url = "http://www.stat.columbia.edu/~gelman/arm/examples/police/frisk_with_noise.dat" 
df = pd.read_csv(url, skiprows=6, delimiter=" ")
df.head()
```
</div>

You should se a table like this:
<center>
  <div markdown="0" style="text-align: right">
    <table class="simpletable">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>stops</th>
          <th>pop</th>
          <th>past.arrests</th>
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
</center>

The data consists of _stop and frisk data_ ( __with noise added to protect confidentiality__ ). This is important because it means that the estimates here will not reproduce the exact same results as in the book or the article. But the lessons of it remain true.
Here's a quick description of the data.

* **stops**: The number of police stops between January 1998 and March 1999, for each combination of precinct, ethnicity and type of crime.
* **pop**: The population.
* **past.arrests**: The number of arrests that took place in 1997 for each combination of precinct, ethnicity and type of crime.
* **precinct**: Index for the precinct (1-75).
* **eth**: Indicator for ethnicity, black (1), hispanic (2), white (3). Other ethnic groups were excluded because ambiguities in the classification would cause large distortions in the analysis[^2].
* **crime**: Indicator for the type,  violent (1), weapons (2), property (3), drug (4).

[^2]: Andrew Gelman, Jeffrey Fagan & Alex Kiss (2007) An Analysis of the New York City Police Department's “Stop-and-Frisk” Policy in the Context of Claims of Racial Bias, Journal of the American Statistical Association, 102:479, 813-823, DOI: 10.1198/016214506000001040

In a Poisson model, each observation corresponds to a setting like a location or a time interval. In this example, the setting is precinct and ethincity -- we index these with the letter $$i$$. The response variable that we want to model, $$y$$, is the number of police stops. Poisson regression is an example of a generalised linear model, so, like in ordinary linear regression or like in logistic regression, we model the variation in $$y$$ with linear predictors $$X$$.

\begin{align}
y_i &\sim \mathrm{Poisson}(\theta_i) \newline
\theta_i &= \exp (X_i \beta) \newline
X_i\beta &= \beta_0 + X_{i,1}\beta_1 + X_{i,2}\beta_2 + ... + X_{i,D}\beta_D .
\end{align}
My notation implicitly assumes that $$X_{i, 0} = 1$$ for all observations, just so that I don't have to write the intercept term separately. While the model above would work just fine, it is most common to model $$y$$ as relative to some baseline variable $$u$$. This baseline variable is also called the _exposure_. So, the model we use is written as 

\begin{align}
y_i \sim \mathrm{Poisson}(u_i \theta_i) = \mathrm{Poisson}(\exp (X_i \beta + \log(u_i))).
\end{align}
So we see that the logarithm of the exposure plays the role of an offset term.

As in the book[^1], we are going to fit the model in 3 different ways. But before that, we need to put our data in the right shape.

<div class="input_area" markdown="1">  

```python
X = (df
    .groupby(['eth', 'precinct'])[["stops", "past.arrests"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['eth', 'precinct'])
    .assign(intercept=1)  # Adds a column called 'intercept' with all values equal to 1.
    .sort_values(by='stops')
    .reset_index(drop=True)
)

y = X.pop("stops")
```
</div>

Pretty neat, huh? I learned the above "trick" from a colleague, who in turn says he learned it from [this](https://tomaugspurger.github.io/method-chaining.html) blog[^3]. Every processing step takes place in a separate line which makes it easier to read, and your code is not cluttered with multiple assignments to `X`. We added the column `intercept` because we will need to pass that explicitly to the `statsmodels.api` (this step would not be necessary if we were using the `statsmodels.formula.api` instead, but I'll not do that here).

[^3]: https://tomaugspurger.github.io/method-chaining.html

## Poisson regression

### Offset and constant term only
First we fit the model without any predictors,
\begin{align}
y_i \sim \mathrm{Poisson}(\exp (\beta_0 + \log(u_i))).
\end{align}


<div class="input_area" markdown="1">

```python
model_no_indicators = sm.GLM(
    y,
    X["intercept"],
    offset=np.log(X["past.arrests"]),
    family=sm.families.Poisson(),
)
result_no_indicators = model_no_indicators.fit()
print(result_no_indicators.summary())
```

</div>
That should print the following output:
<center>
<div class="input_area" markdown="1" style="font-size: 14px;">
```plain
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                  stops   No. Observations:                  225
Model:                            GLM   Df Residuals:                      224
Model Family:                 Poisson   Df Model:                            0
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -23913.
Date:                Wed, 20 May 2020   Deviance:                       46120.
Time:                        08:01:45   Pearson chi2:                 4.96e+04
No. Iterations:                     5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept     -0.5877      0.003   -213.058      0.000      -0.593      -0.582
==============================================================================
```
</div>
</center>

### Ethnicity as a predictor

### Ethicity and precinct as predictors

## Overdispersed Poisson
