---
layout: single
title:  "Chapter 6: Generalised linear models"
date:   2020-05-18
mathjax: true
---


# Generalised linear models


  <div class="input_area" markdown="1">
  
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
  

  <div class="input_area" markdown="1">
  
```python
df2 = (df
    .groupby(['eth', 'precinct'])[["stops", "past_arrests"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['eth', 'precinct'])
    .assign(intercept=1)
    .sort_values(by='stops')
    .reset_index(drop=True)
)
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
f, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].hist(df2.stops)
axes[0].set_xlabel("Stops")
axes[0].set_ylabel("Count")
axes[1].plot(df2.loc[df2.eth_1 == 1].past_arrests, df2.loc[df2.eth_1 == 1].stops, 'o', label="Black")
axes[1].plot(df2.loc[df2.eth_2 == 1].past_arrests, df2.loc[df2.eth_2 == 1].stops, 'o', label="Hispanic")
axes[1].plot(df2.loc[df2.eth_3 == 1].past_arrests, df2.loc[df2.eth_3 == 1].stops, 'o', label="White")
axes[1].set_xlabel("Stops", fontsize=14)
axes[1].set_ylabel("Past arrests", fontsize=14)
# plt.rc('font', size=14)
plt.legend(frameon=False)
plt.show()
```

  </div>
  

![png](/images/blog-images/2020-05-18-generalised_linear_models/police_stops_4_0.png)


## Poisson regression, exposure and overdispersion

This is just a test equation place holder 

$$
\kappa(x_{i+1}\mid x_i) = \mathrm{min}\left(1, \frac{\pi(x_{i+1})q(x_i\mid x_{i+1})}{\pi(x_{i})q(x_{i+1}\mid x_{i})}\right) = \mathrm{min}(1, H),
$$


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
  <th>Date:</th>            <td>Wed, 13 May 2020</td> <th>  Deviance:          </th> <td>  46120.</td>
</tr>
<tr>
  <th>Time:</th>                <td>06:02:50</td>     <th>  Pearson chi2:      </th> <td>4.96e+04</td>
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
  <th>Date:</th>            <td>Wed, 13 May 2020</td> <th>  Deviance:          </th> <td>  45437.</td>
</tr>
<tr>
  <th>Time:</th>                <td>06:03:53</td>     <th>  Pearson chi2:      </th> <td>4.94e+04</td>
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
  

  <div class="input_area" markdown="1">
  
```python
f = plt.figure(figsize=(6 * 1.618, 6))
plt.errorbar(
    *group_residuals(df2.stops, result_no_indicators.fittedvalues, offset=-0.15),
    label='No indicators',
    marker='o',
    linestyle=''
)
plt.errorbar(
    *group_residuals(df2.stops, result_with_ethnicity.fittedvalues),
    label='With ethnicity',
    marker='o',
    linestyle=''
)
plt.errorbar(
    *group_residuals(df2.stops, result_with_ethnicity_and_precinct.fittedvalues, offset=0.15),
    label='With ethnicity and precinct',
    marker='o',
    linestyle=''
)

plt.xlabel("Quantile", fontsize=14)
plt.ylabel("Residual", fontsize=14)
plt.legend(loc='upper left')
plt.show()
```

  </div>
  

![png](/images/blog-images/2020-05-18-generalised_linear_models/police_stops_14_0.png)



  <div class="input_area" markdown="1">
  
```python
f, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].plot(
    df2.stops,
    df2.stops - result_with_ethnicity_and_precinct.fittedvalues,
    marker='.',
    linestyle=''
)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_ylabel("Residual", fontsize=14)
axes[0].set_xlabel("Stops", fontsize=14)

axes[1].plot(
    df2.stops,
    (df2.stops - result_with_ethnicity_and_precinct.fittedvalues) / np.sqrt(result_with_ethnicity_and_precinct.fittedvalues),
    marker='.',
    linestyle=''
)

axes[1].axhline(y=-2, linestyle=':', color='black', label="$\pm 2\sigma$")
axes[1].axhline(y=+2, linestyle=':', color='black',)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_ylabel("Standardized Residual", fontsize=14)
axes[1].set_xlabel("Stops", fontsize=14)
axes[1].legend()


plt.show()
```

  </div>
  

![png](/images/blog-images/2020-05-18-generalised_linear_models/police_stops_15_0.png)


The standardised residuals are already stored in the fitted model in the attribute `resid_pearson`, so we don't need to compute these by hand.


  <div class="input_area" markdown="1">
  
```python
z_residuals = (df2.stops - result_with_ethnicity_and_precinct.fittedvalues) / np.sqrt(result_with_ethnicity_and_precinct.fittedvalues)
(z_residuals == result_with_ethnicity_and_precinct.resid_pearson).all()
```

  </div>
  



  {:.output_data_text}</p>

<pre><code>  True</code></pre>
<p>



  <div class="input_area" markdown="1">
  
```python
overdispersion_ratio = sum(result_with_ethnicity_and_precinct.resid_pearson ** 2) / result_with_ethnicity_and_precinct.df_resid
overdispersion_test = scipy.stats.chisquare(
    result_with_ethnicity_and_precinct.resid_pearson,
    ddof=result_with_ethnicity_and_precinct.df_resid,
)
```

  </div>
  

  <div class="input_area" markdown="1">
  
```python
print(f"Overdispersion ratio is {overdispersion_ratio:.2f}")
print(f"p-value of overdispersion test is {overdispersion_test.pvalue:.2f}")
```

  </div>
  
  {:.output_stream}</p>

<pre><code>  Overdispersion ratio is 21.24
p-value of overdispersion test is 1.00
</code></pre>
<p>