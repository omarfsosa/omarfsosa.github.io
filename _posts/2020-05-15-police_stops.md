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
  

![png](police_stops_files/police_stops_4_0.png)


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
result_with_ethnicity_and_precinct.summary()
```

  </div>
  



  <div markdown="0">
  <table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>stops</td>      <th>  No. Observations:  </th>  <td>   225</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   147</td> 
</tr>
<tr>
  <th>Model Family:</th>         <td>Poisson</td>     <th>  Df Model:          </th>  <td>    77</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -2449.0</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 13 May 2020</td> <th>  Deviance:          </th> <td>  3191.4</td>
</tr>
<tr>
  <th>Time:</th>                <td>06:27:03</td>     <th>  Pearson chi2:      </th> <td>3.12e+03</td>
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
        <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>past_arrests</th> <td>  6.63e-05</td> <td> 4.38e-06</td> <td>   15.131</td> <td> 0.000</td> <td> 5.77e-05</td> <td> 7.49e-05</td>
</tr>
<tr>
  <th>eth_2</th>        <td>    0.0410</td> <td>    0.007</td> <td>    5.714</td> <td> 0.000</td> <td>    0.027</td> <td>    0.055</td>
</tr>
<tr>
  <th>eth_3</th>        <td>   -0.3595</td> <td>    0.010</td> <td>  -34.996</td> <td> 0.000</td> <td>   -0.380</td> <td>   -0.339</td>
</tr>
<tr>
  <th>precinct_2</th>   <td>   -0.1467</td> <td>    0.074</td> <td>   -1.981</td> <td> 0.048</td> <td>   -0.292</td> <td>   -0.002</td>
</tr>
<tr>
  <th>precinct_3</th>   <td>    0.4934</td> <td>    0.057</td> <td>    8.667</td> <td> 0.000</td> <td>    0.382</td> <td>    0.605</td>
</tr>
<tr>
  <th>precinct_4</th>   <td>    1.2023</td> <td>    0.058</td> <td>   20.890</td> <td> 0.000</td> <td>    1.090</td> <td>    1.315</td>
</tr>
<tr>
  <th>precinct_5</th>   <td>    0.2021</td> <td>    0.057</td> <td>    3.543</td> <td> 0.000</td> <td>    0.090</td> <td>    0.314</td>
</tr>
<tr>
  <th>precinct_6</th>   <td>    1.1458</td> <td>    0.058</td> <td>   19.738</td> <td> 0.000</td> <td>    1.032</td> <td>    1.260</td>
</tr>
<tr>
  <th>precinct_7</th>   <td>    0.2066</td> <td>    0.064</td> <td>    3.212</td> <td> 0.001</td> <td>    0.081</td> <td>    0.333</td>
</tr>
<tr>
  <th>precinct_8</th>   <td>   -0.6328</td> <td>    0.059</td> <td>  -10.698</td> <td> 0.000</td> <td>   -0.749</td> <td>   -0.517</td>
</tr>
<tr>
  <th>precinct_9</th>   <td>    0.5091</td> <td>    0.078</td> <td>    6.512</td> <td> 0.000</td> <td>    0.356</td> <td>    0.662</td>
</tr>
<tr>
  <th>precinct_10</th>  <td>    0.3819</td> <td>    0.059</td> <td>    6.485</td> <td> 0.000</td> <td>    0.266</td> <td>    0.497</td>
</tr>
<tr>
  <th>precinct_11</th>  <td>    0.6497</td> <td>    0.062</td> <td>   10.549</td> <td> 0.000</td> <td>    0.529</td> <td>    0.770</td>
</tr>
<tr>
  <th>precinct_12</th>  <td>    1.1638</td> <td>    0.061</td> <td>   19.054</td> <td> 0.000</td> <td>    1.044</td> <td>    1.283</td>
</tr>
<tr>
  <th>precinct_13</th>  <td>    1.0087</td> <td>    0.055</td> <td>   18.230</td> <td> 0.000</td> <td>    0.900</td> <td>    1.117</td>
</tr>
<tr>
  <th>precinct_14</th>  <td>    0.5892</td> <td>    0.059</td> <td>   10.015</td> <td> 0.000</td> <td>    0.474</td> <td>    0.704</td>
</tr>
<tr>
  <th>precinct_15</th>  <td>    1.0003</td> <td>    0.055</td> <td>   18.162</td> <td> 0.000</td> <td>    0.892</td> <td>    1.108</td>
</tr>
<tr>
  <th>precinct_16</th>  <td>    0.7853</td> <td>    0.060</td> <td>   13.024</td> <td> 0.000</td> <td>    0.667</td> <td>    0.903</td>
</tr>
<tr>
  <th>precinct_17</th>  <td>   -0.2003</td> <td>    0.061</td> <td>   -3.266</td> <td> 0.001</td> <td>   -0.320</td> <td>   -0.080</td>
</tr>
<tr>
  <th>precinct_18</th>  <td>   -0.0922</td> <td>    0.057</td> <td>   -1.618</td> <td> 0.106</td> <td>   -0.204</td> <td>    0.019</td>
</tr>
<tr>
  <th>precinct_19</th>  <td>    0.1654</td> <td>    0.059</td> <td>    2.808</td> <td> 0.005</td> <td>    0.050</td> <td>    0.281</td>
</tr>
<tr>
  <th>precinct_20</th>  <td>   -0.1762</td> <td>    0.058</td> <td>   -3.056</td> <td> 0.002</td> <td>   -0.289</td> <td>   -0.063</td>
</tr>
<tr>
  <th>precinct_21</th>  <td>    0.3053</td> <td>    0.057</td> <td>    5.332</td> <td> 0.000</td> <td>    0.193</td> <td>    0.417</td>
</tr>
<tr>
  <th>precinct_22</th>  <td>    1.1046</td> <td>    0.054</td> <td>   20.503</td> <td> 0.000</td> <td>    0.999</td> <td>    1.210</td>
</tr>
<tr>
  <th>precinct_23</th>  <td>    0.4883</td> <td>    0.056</td> <td>    8.785</td> <td> 0.000</td> <td>    0.379</td> <td>    0.597</td>
</tr>
<tr>
  <th>precinct_24</th>  <td>    1.2801</td> <td>    0.055</td> <td>   23.418</td> <td> 0.000</td> <td>    1.173</td> <td>    1.387</td>
</tr>
<tr>
  <th>precinct_25</th>  <td>    0.7828</td> <td>    0.054</td> <td>   14.419</td> <td> 0.000</td> <td>    0.676</td> <td>    0.889</td>
</tr>
<tr>
  <th>precinct_26</th>  <td>   -0.3243</td> <td>    0.058</td> <td>   -5.572</td> <td> 0.000</td> <td>   -0.438</td> <td>   -0.210</td>
</tr>
<tr>
  <th>precinct_27</th>  <td>    1.9055</td> <td>    0.056</td> <td>   34.228</td> <td> 0.000</td> <td>    1.796</td> <td>    2.015</td>
</tr>
<tr>
  <th>precinct_28</th>  <td>   -0.9594</td> <td>    0.062</td> <td>  -15.407</td> <td> 0.000</td> <td>   -1.081</td> <td>   -0.837</td>
</tr>
<tr>
  <th>precinct_29</th>  <td>    1.0177</td> <td>    0.055</td> <td>   18.592</td> <td> 0.000</td> <td>    0.910</td> <td>    1.125</td>
</tr>
<tr>
  <th>precinct_30</th>  <td>    0.4139</td> <td>    0.056</td> <td>    7.409</td> <td> 0.000</td> <td>    0.304</td> <td>    0.523</td>
</tr>
<tr>
  <th>precinct_31</th>  <td>    1.6544</td> <td>    0.056</td> <td>   29.453</td> <td> 0.000</td> <td>    1.544</td> <td>    1.765</td>
</tr>
<tr>
  <th>precinct_32</th>  <td>    1.4007</td> <td>    0.060</td> <td>   23.466</td> <td> 0.000</td> <td>    1.284</td> <td>    1.518</td>
</tr>
<tr>
  <th>precinct_33</th>  <td>    1.0102</td> <td>    0.054</td> <td>   18.557</td> <td> 0.000</td> <td>    0.903</td> <td>    1.117</td>
</tr>
<tr>
  <th>precinct_34</th>  <td>    1.4987</td> <td>    0.054</td> <td>   27.564</td> <td> 0.000</td> <td>    1.392</td> <td>    1.605</td>
</tr>
<tr>
  <th>precinct_35</th>  <td>    0.8572</td> <td>    0.063</td> <td>   13.558</td> <td> 0.000</td> <td>    0.733</td> <td>    0.981</td>
</tr>
<tr>
  <th>precinct_36</th>  <td>    1.5975</td> <td>    0.059</td> <td>   27.157</td> <td> 0.000</td> <td>    1.482</td> <td>    1.713</td>
</tr>
<tr>
  <th>precinct_37</th>  <td>    1.4233</td> <td>    0.060</td> <td>   23.715</td> <td> 0.000</td> <td>    1.306</td> <td>    1.541</td>
</tr>
<tr>
  <th>precinct_38</th>  <td>    1.7626</td> <td>    0.056</td> <td>   31.363</td> <td> 0.000</td> <td>    1.652</td> <td>    1.873</td>
</tr>
<tr>
  <th>precinct_39</th>  <td>    0.2051</td> <td>    0.058</td> <td>    3.519</td> <td> 0.000</td> <td>    0.091</td> <td>    0.319</td>
</tr>
<tr>
  <th>precinct_40</th>  <td>    1.5250</td> <td>    0.061</td> <td>   24.805</td> <td> 0.000</td> <td>    1.405</td> <td>    1.645</td>
</tr>
<tr>
  <th>precinct_41</th>  <td>    1.9162</td> <td>    0.056</td> <td>   34.391</td> <td> 0.000</td> <td>    1.807</td> <td>    2.025</td>
</tr>
<tr>
  <th>precinct_42</th>  <td>    0.9928</td> <td>    0.055</td> <td>   18.099</td> <td> 0.000</td> <td>    0.885</td> <td>    1.100</td>
</tr>
<tr>
  <th>precinct_43</th>  <td>    0.3925</td> <td>    0.058</td> <td>    6.824</td> <td> 0.000</td> <td>    0.280</td> <td>    0.505</td>
</tr>
<tr>
  <th>precinct_44</th>  <td>    0.8002</td> <td>    0.057</td> <td>   14.051</td> <td> 0.000</td> <td>    0.689</td> <td>    0.912</td>
</tr>
<tr>
  <th>precinct_45</th>  <td>    0.6253</td> <td>    0.056</td> <td>   11.152</td> <td> 0.000</td> <td>    0.515</td> <td>    0.735</td>
</tr>
<tr>
  <th>precinct_46</th>  <td>    0.4724</td> <td>    0.055</td> <td>    8.569</td> <td> 0.000</td> <td>    0.364</td> <td>    0.580</td>
</tr>
<tr>
  <th>precinct_47</th>  <td>    1.1393</td> <td>    0.060</td> <td>   18.992</td> <td> 0.000</td> <td>    1.022</td> <td>    1.257</td>
</tr>
<tr>
  <th>precinct_48</th>  <td>    0.2252</td> <td>    0.058</td> <td>    3.866</td> <td> 0.000</td> <td>    0.111</td> <td>    0.339</td>
</tr>
<tr>
  <th>precinct_49</th>  <td>    1.1064</td> <td>    0.060</td> <td>   18.536</td> <td> 0.000</td> <td>    0.989</td> <td>    1.223</td>
</tr>
<tr>
  <th>precinct_50</th>  <td>    0.6770</td> <td>    0.056</td> <td>   12.137</td> <td> 0.000</td> <td>    0.568</td> <td>    0.786</td>
</tr>
<tr>
  <th>precinct_51</th>  <td>    0.2334</td> <td>    0.059</td> <td>    3.952</td> <td> 0.000</td> <td>    0.118</td> <td>    0.349</td>
</tr>
<tr>
  <th>precinct_52</th>  <td>    0.3794</td> <td>    0.056</td> <td>    6.779</td> <td> 0.000</td> <td>    0.270</td> <td>    0.489</td>
</tr>
<tr>
  <th>precinct_53</th>  <td>    0.5494</td> <td>    0.059</td> <td>    9.378</td> <td> 0.000</td> <td>    0.435</td> <td>    0.664</td>
</tr>
<tr>
  <th>precinct_54</th>  <td>    0.4318</td> <td>    0.060</td> <td>    7.175</td> <td> 0.000</td> <td>    0.314</td> <td>    0.550</td>
</tr>
<tr>
  <th>precinct_55</th>  <td>    0.4430</td> <td>    0.059</td> <td>    7.490</td> <td> 0.000</td> <td>    0.327</td> <td>    0.559</td>
</tr>
<tr>
  <th>precinct_56</th>  <td>    0.8961</td> <td>    0.066</td> <td>   13.580</td> <td> 0.000</td> <td>    0.767</td> <td>    1.025</td>
</tr>
<tr>
  <th>precinct_57</th>  <td>    1.5571</td> <td>    0.061</td> <td>   25.365</td> <td> 0.000</td> <td>    1.437</td> <td>    1.677</td>
</tr>
<tr>
  <th>precinct_58</th>  <td>    1.5390</td> <td>    0.055</td> <td>   28.189</td> <td> 0.000</td> <td>    1.432</td> <td>    1.646</td>
</tr>
<tr>
  <th>precinct_59</th>  <td>    1.0860</td> <td>    0.059</td> <td>   18.431</td> <td> 0.000</td> <td>    0.971</td> <td>    1.201</td>
</tr>
<tr>
  <th>precinct_60</th>  <td>    0.6519</td> <td>    0.056</td> <td>   11.730</td> <td> 0.000</td> <td>    0.543</td> <td>    0.761</td>
</tr>
<tr>
  <th>precinct_61</th>  <td>    1.2562</td> <td>    0.057</td> <td>   22.124</td> <td> 0.000</td> <td>    1.145</td> <td>    1.367</td>
</tr>
<tr>
  <th>precinct_62</th>  <td>    1.1686</td> <td>    0.057</td> <td>   20.602</td> <td> 0.000</td> <td>    1.057</td> <td>    1.280</td>
</tr>
<tr>
  <th>precinct_63</th>  <td>    1.3177</td> <td>    0.058</td> <td>   22.600</td> <td> 0.000</td> <td>    1.203</td> <td>    1.432</td>
</tr>
<tr>
  <th>precinct_64</th>  <td>    1.7313</td> <td>    0.058</td> <td>   29.961</td> <td> 0.000</td> <td>    1.618</td> <td>    1.845</td>
</tr>
<tr>
  <th>precinct_65</th>  <td>    1.8939</td> <td>    0.055</td> <td>   34.333</td> <td> 0.000</td> <td>    1.786</td> <td>    2.002</td>
</tr>
<tr>
  <th>precinct_66</th>  <td>    2.1038</td> <td>    0.054</td> <td>   38.792</td> <td> 0.000</td> <td>    1.998</td> <td>    2.210</td>
</tr>
<tr>
  <th>precinct_67</th>  <td>    1.1323</td> <td>    0.055</td> <td>   20.522</td> <td> 0.000</td> <td>    1.024</td> <td>    1.240</td>
</tr>
<tr>
  <th>precinct_68</th>  <td>    2.2126</td> <td>    0.059</td> <td>   37.338</td> <td> 0.000</td> <td>    2.096</td> <td>    2.329</td>
</tr>
<tr>
  <th>precinct_69</th>  <td>    1.7336</td> <td>    0.058</td> <td>   29.767</td> <td> 0.000</td> <td>    1.619</td> <td>    1.848</td>
</tr>
<tr>
  <th>precinct_70</th>  <td>    0.9009</td> <td>    0.056</td> <td>   16.230</td> <td> 0.000</td> <td>    0.792</td> <td>    1.010</td>
</tr>
<tr>
  <th>precinct_71</th>  <td>    1.4523</td> <td>    0.054</td> <td>   26.760</td> <td> 0.000</td> <td>    1.346</td> <td>    1.559</td>
</tr>
<tr>
  <th>precinct_72</th>  <td>    1.3916</td> <td>    0.054</td> <td>   25.784</td> <td> 0.000</td> <td>    1.286</td> <td>    1.497</td>
</tr>
<tr>
  <th>precinct_73</th>  <td>    0.8628</td> <td>    0.054</td> <td>   15.899</td> <td> 0.000</td> <td>    0.756</td> <td>    0.969</td>
</tr>
<tr>
  <th>precinct_74</th>  <td>    1.1068</td> <td>    0.058</td> <td>   19.044</td> <td> 0.000</td> <td>    0.993</td> <td>    1.221</td>
</tr>
<tr>
  <th>precinct_75</th>  <td>    1.5687</td> <td>    0.076</td> <td>   20.712</td> <td> 0.000</td> <td>    1.420</td> <td>    1.717</td>
</tr>
<tr>
  <th>intercept</th>    <td>   -1.4443</td> <td>    0.051</td> <td>  -28.208</td> <td> 0.000</td> <td>   -1.545</td> <td>   -1.344</td>
</tr>
</table>
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
  

![png](police_stops_files/police_stops_14_0.png)



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
  

![png](police_stops_files/police_stops_15_0.png)


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