# Overview
This report distinguishes itself by its emphasis on feature visualization, selection techniques and  that set it apart from other projects. It employs various methods for feature selection, including correlation-based selection, univariate feature selection, recursive feature elimination, and recursive feature elimination with cross-validation Additionally, Principal Component Analysis is employed to gain insights into the optimal number of components. 

The next section, employ different machine learning techniques to predicts if there are a breast cancer or not.



```python
import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
```


```python
bc_data = pd.read_csv("/Users/felixyh/Documents/Data Analytics/Cancer Prediction/BC_data.csv")
```


```python
bc_data.head()
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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
bc_data = bc_data.drop(columns={"id","Unnamed: 32"})

```


```python
bc_data.head()
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
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



# Data Description 


| Variable Name      | Role    | Type        | Description                                                       | Missing Values |
|--------------------|---------|-------------|-------------------------------------------------------------------|----------------|
| ID                 | ID      | Categorical | ID number                                                         | FALSE          |
| Diagnosis          | Target  | Categorical | Diagnosis (M = malignant, B = benign)                             | FALSE          |
| radius1            | Feature | Continuous  | radius (mean of distances from center to points on the perimeter) | FALSE          |
| texture1           | Feature | Continuous  | texture (standard deviation of gray-scale values)                 | FALSE          |
| perimeter1         | Feature | Continuous  | perimeter                                                         | FALSE          |
| area1              | Feature | Continuous  | area                                                              | FALSE          |
| smoothness1        | Feature | Continuous  | smoothness (local variation in radius lengths)                    | FALSE          |
| compactness1       | Feature | Continuous  | compactness (perimeter^2 / area - 1.0)                            | FALSE          |
| concavity1         | Feature | Continuous  | concavity (severity of concave portions of the contour)           | FALSE          |
| concave_points1    | Feature | Continuous  | concave points (number of concave portions of the contour)        | FALSE          |
| symmetry1          | Feature | Continuous  | symmetry                                                          | FALSE          |
| fractal_dimension1 | Feature | Continuous  | fractal dimension ("coastline approximation" - 1)                 | FALSE          |
| radius2            | Feature | Continuous  |                                                                   | FALSE          |
| texture2           | Feature | Continuous  |                                                                   | FALSE          |
| perimeter2         | Feature | Continuous  |                                                                   | FALSE          |
| area2              | Feature | Continuous  |                                                                   | FALSE          |
| smoothness2        | Feature | Continuous  |                                                                   | FALSE          |
| compactness2       | Feature | Continuous  |                                                                   | FALSE          |
| concavity2         | Feature | Continuous  |                                                                   | FALSE          |
| concave_points2    | Feature | Continuous  |                                                                   | FALSE          |
| symmetry2          | Feature | Continuous  |                                                                   | FALSE          |
| fractal_dimension2 | Feature | Continuous  |                                                                   | FALSE          |
| radius3            | Feature | Continuous  |                                                                   | FALSE          |
| texture3           | Feature | Continuous  |                                                                   | FALSE          |
| perimeter3         | Feature | Continuous  |                                                                   | FALSE          |
| area3              | Feature | Continuous  |                                                                   | FALSE          |
| smoothness3        | Feature | Continuous  |                                                                   | FALSE          |
| compactness3       | Feature | Continuous  |                                                                   | FALSE          |
| concavity3         | Feature | Continuous  |                                                                   | FALSE          |
| concave_points3    | Feature | Continuous  |                                                                   | FALSE          |
| symmetry3          | Feature | Continuous  |                                                                   | FALSE          |
| fractal_dimension3 | Feature | Continuous  |                                                                   | FALSE          |

# Exploratory Data Analysis


```python
bc_data.columns
```




    Index(['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'],
          dtype='object')




```python
def count_plot(df, xvar, huevar=None, color=0, palette=None, order=None):
    ''' 
    This function takes a variable(dataframe),
    convert and display the count of variable and also
    plot the  count distribution of the variable
    '''
    # set plot dimensions 
    plt.figure(figsize = [14,6])
    #plot
    sns.countplot(data=df, x = xvar, hue=huevar, color=sns.color_palette()[color],palette=palette, order=order, edgecolor='black')
    # display proportions
    if huevar:
        display(df.groupby(xvar)[huevar].value_counts(normalize=True).mul(100).round(2).unstack().T)
    else:
        display(df[xvar].value_counts(normalize=True).mul(100).round(2).to_frame().T)
    # clean up variables names
    xvar=xvar.replace("_"," ") # replace _ with a space
    if huevar:
        huevar=huevar.replace("_"," ") 
    # Add title and format it
    plt.title(f'''Distribution of {xvar} {'by' if huevar else ''} {huevar if huevar else ''}'''.title(), fontsize = 14, weight = "bold")
    # Add x label and format it
    plt.xlabel(xvar.title(), fontsize = 10, weight = 'bold')
    # Add y label and format it
    plt.ylabel('Frequency'.title(), fontsize = 10, weight = "bold");
```


```python
count_plot(bc_data,'diagnosis')
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
      <th>diagnosis</th>
      <th>B</th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>proportion</th>
      <td>62.74</td>
      <td>37.26</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](CancerML_files/CancerML_10_1.png)
    




The "Number of Benign" tumors were 357, which accounts for 62.74% of the total cases. 

The "Number of Malignant" tumors were 212, which accounts for 37.26% of the total cases. 




```python

feature = bc_data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

feature.describe()
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
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>



## Some Vizualizations 

#### Univariate & Bivariate


```python
Target = bc_data["diagnosis"]
data = feature
data_n_2 = (data - data.mean()) / (data.std())             
data = pd.concat([Target,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=False, inner="quart")
plt.xticks(rotation=90);
```


    
![png](CancerML_files/CancerML_16_0.png)
    


Texture Mean Feature:
In the case of the "texture_mean" feature, we can see that there is a noticeable separation between the median values of the Malignant and Benign groups.
This separation suggests that the "texture_mean" feature could be valuable for classification because it appears to have discriminative power.It could help differentiate between Malignant and Benign cases. When a feature shows clear separation between different classes, it is often a good indicator for classification tasks.

Fractal Dimension Mean Feature:
On the other hand, when looking at the "fractal_dimension_mean" feature, we observe that the medians of the Malignant and Benign groups are closer together, and there is less noticeable separation.
This lack of separation suggests that the "fractal_dimension_mean" feature may not provide strong discriminatory information for classification. In such cases, this feature might not be as useful for distinguishing between Malignant and Benign cases.

A similar interpretation holds for the other features 

Violin plot for the remaining feature


```python
Target = bc_data["diagnosis"]
data = feature
data_n_2 = (data - data.mean()) / (data.std())             
data = pd.concat([Target,data_n_2.iloc[:,10:]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=False, inner="quart")
plt.xticks(rotation=90);
```


    
![png](CancerML_files/CancerML_19_0.png)
    


# Feature Selection

### 1. Using Correlation




```python
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(feature.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```




    <Axes: >




    
![png](CancerML_files/CancerML_22_1.png)
    


Since the target variable is categorical , we need to first convert it


```python
bc_data2 = bc_data.copy()
bc_data2['diagnosis'] = bc_data['diagnosis'].map({'B':0,'M':1})
#Compute the correlation matrix
correlation_matrix = bc_data2.corr()

# Set a correlation threshold
threshold = 0.9  # Adjust this threshold as needed

# Identify highly correlated features
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# Print the highly correlated features
print("Highly Correlated Features:", correlated_features)
```

    Highly Correlated Features: {'perimeter_worst', 'texture_worst', 'perimeter_se', 'perimeter_mean', 'radius_worst', 'area_worst', 'concave points_worst', 'area_mean', 'concave points_mean', 'area_se'}


### 2. Using Random Forest


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Split the data into features and target
X = bc_data2.drop(columns=['diagnosis'])  # Features
y = bc_data2['diagnosis']  # Categorical target variable

# Apply the Chi-Square test for feature selection
selector = SelectKBest(chi2, k=3)  # Select the top 3 features
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_indices]
print("Selected Features:", selected_features)

```

    Selected Features: Index(['area_mean', 'area_se', 'area_worst'], dtype='object')


area_mean', 'area_se',  and 'area_worst'. These three features are considered the most informative.

### 3. Using Recursive feature elimination (RFE) with random forest


```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)

# Create an RFE selector
rfe_selector = RFE(estimator=rf_classifier, n_features_to_select=5, step=1)

# Fit the RFE selector to your data
rfe_selector = rfe_selector.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe_selector.support_]

# Print the selected features
print("Selected Features:", selected_features.tolist())
```

    Selected Features: ['concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']


'concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', and 'concave points_worst'. This indicates that, after applying the Recursive Feature Elimination (RFE) technique with a Random Forest classifier to the dataset, these five features have been identified as the most important or informative.

### 4. Using Recursive feature elimination -- cross validation and random forest classification


```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)

# Create an RFECV selector with cross-validation
rfecv_selector = RFECV(estimator=rf_classifier, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# Fit the RFECV selector to your data
rfecv_selector = rfecv_selector.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfecv_selector.support_]

# Print the selected features
print('Optimal number of features :', rfecv_selector.n_features_)
print("Selected Features:", selected_features.tolist())

```

    Optimal number of features : 14
    Selected Features: ['texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_se', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'concavity_worst', 'concave points_worst']


### 5. Using PCA for Feature Selection 


```python
from sklearn.decomposition import PCA

# Create a PCA object with the desired number of components
n_components = 3  
pca = PCA(n_components=n_components)

# Fit PCA on your feature matrix
X_pca = pca.fit_transform(X_train)

# X_pca now contains the lower-dimensional representation of your data

# You can access the principal components (eigenvectors) and their explained variance
explained_variance = pca.explained_variance_ratio_
components = pca.components_

# Print the explained variance for each component
print("Explained Variance Ratios:")
print(explained_variance)

# Print the principal components (eigenvectors)
print("Principal Components:")
print(components)

```

    Explained Variance Ratios:
    [0.98135117 0.01652016 0.00191059]
    Principal Components:
    [[ 5.06812127e-03  2.00176762e-03  3.48376453e-02  5.23452415e-01
       3.64311374e-06  3.56976167e-05  7.73235570e-05  4.57185961e-05
       7.26975184e-06 -3.11146853e-06  3.22331624e-04 -4.01140585e-05
       2.25789791e-03  5.81690902e-02 -7.08544186e-07  4.26443004e-06
       7.98447586e-06  2.95091420e-06 -7.62162344e-07 -1.89845423e-07
       7.04081093e-03  2.81731434e-03  4.82672469e-02  8.47925754e-01
       5.55537931e-06  8.43962732e-05  1.54869697e-04  6.90191428e-05
       1.79157621e-05  1.08268771e-07]
     [ 8.90440410e-03 -3.47219268e-03  5.99175338e-02  8.47097129e-01
      -1.79747265e-05 -1.26469855e-05  6.60379629e-05  3.68197163e-05
      -2.14510439e-05 -1.64349299e-05  3.20767950e-05  3.96646833e-04
       1.50709143e-03  2.75374998e-02  2.32685350e-06  1.12577500e-05
       3.11948753e-05  7.39152527e-06  1.56848363e-05  1.67529466e-07
      -1.25246394e-03 -1.36604427e-02 -4.18161146e-03 -5.27046894e-01
      -8.25803421e-05 -2.79111517e-04 -2.08269597e-04 -5.36749479e-05
      -1.47973474e-04 -5.60167855e-05]
     [-1.21114418e-02 -2.19680543e-03 -7.04743794e-02 -4.71734634e-02
       6.72013307e-05  9.46011612e-05  2.30569866e-04  1.44004845e-05
       9.65775473e-05  4.82470973e-05  5.79253167e-03  5.28955609e-03
       4.25632539e-02  9.90637881e-01  3.79830820e-05  1.10801323e-04
       1.85240678e-04  3.97181275e-05  8.81683403e-05  2.08868157e-05
      -1.48365037e-02 -2.38083606e-02 -8.76578864e-02 -3.07875501e-02
      -3.00804414e-05 -5.84457784e-04 -6.92402225e-04 -3.02965439e-04
      -3.87977514e-04 -2.32391075e-05]]



```python
import matplotlib.pyplot as plt

# Assuming you've already performed PCA and have explained_variance available

# Create a line plot for explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Explained Variance by Principal Component")
plt.grid(True)
plt.show()
```


    
![png](CancerML_files/CancerML_35_0.png)
    


# SECTION TWO


```python
bc_data3 = bc_data2[['diagnosis','texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_se', 'area_se', 'radius_worst', 'texture_worst', 
                     'perimeter_worst', 'area_worst', 'smoothness_worst', 'concavity_worst', 'concave points_worst']]

# Using the 14 features from the CV feature selection 

# Split the data into features and target
X = bc_data3.drop(columns=['diagnosis'])  # Features
y = bc_data3['diagnosis']  # Categorical target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
import pandas as pd
from sklearn.pipeline import Pipeline

# Load the Iris dataset
bc_data3 = bc_data2[['diagnosis','texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_se', 'area_se', 'radius_worst', 'texture_worst', 
                     'perimeter_worst', 'area_worst', 'smoothness_worst', 'concavity_worst', 'concave points_worst']]

# Using the 14 features from the CV feature selection 

# Split the data into features and target
X = bc_data3.drop(columns=['diagnosis'])  # Features
Y = bc_data3['diagnosis'] # Categorical target variable


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define a list of classifiers, including Logistic Regression
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC(probability=True)),  # Enable probability for AUC
    ('k-Nearest Neighbors', KNeighborsClassifier(n_neighbors=3)),
    ('Naive Bayes', GaussianNB()),
    ('Logistic Regression', LogisticRegression())
]

# Create a list to store results DataFrames
results_dfs = []

# Define a scoring function (weighted F1 score) for cross-validation
scorer = make_scorer(f1_score, average='weighted')

# Iterate over classifiers, train, cross-validate, and evaluate
for name, classifier in classifiers:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('clf', classifier)
    ])
    pipeline.fit(X_train, y_train)

    # Perform k-fold cross-validation (e.g., 5-fold)
    cross_val_scores = cross_val_score(pipeline, X, Y, cv=5, scoring=scorer)

    # Compute the average cross-validation score
    avg_cv_score = cross_val_scores.mean()

    # Make predictions on the test set for additional metrics
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr', average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Create a DataFrame for the current classifier's results
    classifier_results = pd.DataFrame({
        'Modelname': [name],
        'Accuracy': [accuracy],
        # 'AUC': [auc]
        'Recall': [recall],
        'Precesion': [precision],
        'F1': [f1],
        'Cross-Val F1 Score': [avg_cv_score]
    })

    results_dfs.append(classifier_results)

# Concatenate the results DataFrames into a single DataFrame
results_df = pd.concat(results_dfs, ignore_index=True)

# Print the results
results_df

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
      <th>Modelname</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precesion</th>
      <th>F1</th>
      <th>Cross-Val F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Tree</td>
      <td>0.953216</td>
      <td>0.953216</td>
      <td>0.953785</td>
      <td>0.953363</td>
      <td>0.921335</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.970760</td>
      <td>0.970760</td>
      <td>0.971100</td>
      <td>0.970604</td>
      <td>0.963023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Support Vector Machine</td>
      <td>0.982456</td>
      <td>0.982456</td>
      <td>0.982930</td>
      <td>0.982362</td>
      <td>0.971856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>k-Nearest Neighbors</td>
      <td>0.947368</td>
      <td>0.947368</td>
      <td>0.947607</td>
      <td>0.947453</td>
      <td>0.973525</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>0.953216</td>
      <td>0.953216</td>
      <td>0.953173</td>
      <td>0.953054</td>
      <td>0.946996</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Logistic Regression</td>
      <td>0.976608</td>
      <td>0.976608</td>
      <td>0.976608</td>
      <td>0.976608</td>
      <td>0.970032</td>
    </tr>
  </tbody>
</table>
</div>



Decision Tree: This model has an accuracy of approximately 95.3%, indicating that it correctly classifies about 95.3% of the cases. It has high recall (95.3%), suggesting it is effective at identifying malignant tumors. The precision (95.4%) indicates that it makes few false-positive predictions. The F1 score (95.3%) suggests a good balance between precision and recall. Cross-validation F1 score (92.1%) indicates consistent performance.

Random Forest: This ensemble model has a higher accuracy of approximately 97.1%, indicating strong overall performance. It also has high recall (97.1%) and precision (97.1%), showing effectiveness in both identifying malignant tumors and avoiding false positives. The F1 score (97.1%) and cross-validation F1 score (96.3%) are high, indicating excellent performance.

Support Vector Machine: SVM shows even better performance with an accuracy of 98.2%. It has high recall (98.2%) and precision (98.3%), making it effective for diagnosis. Both the F1 score (98.2%) and cross-validation F1 score (97.2%) are high, indicating strong and consistent performance.

k-Nearest Neighbors: KNN has an accuracy of approximately 94.7%, a recall of 94.7%, and a precision of 94.8%. It shows good performance in identifying malignant tumors while maintaining a low rate of false positives. The F1 score (94.7%) and cross-validation F1 score (97.4%) suggest good overall performance.

Naive Bayes: Naive Bayes has an accuracy of approximately 95.3%, similar to Decision Trees. It has high recall (95.3%) and a precision of 95.3%, indicating effectiveness in diagnosing malignant tumors. The F1 score (95.3%) and cross-validation F1 score (94.7%) suggest good balance and consistent performance.

Logistic Regression: Logistic Regression shows an accuracy of approximately 97.7%. It has high recall (97.7%) and precision (97.7%), making it effective in diagnosing malignant tumors. The F1 score (97.7%) and cross-validation F1 score (97.0%) indicate strong and consistent performance.
