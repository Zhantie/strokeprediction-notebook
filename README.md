## Import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import joblib

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

# Data Import en Opschoning

- De dataset wordt ingeladen met `pandas`.
- De `id`-kolom is verwijderd omdat deze niet relevant is.
- Rijen met ontbrekende waarden zijn verwijderd.


```python
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.drop(columns=['id'])
df.dropna(inplace=True)
```

De grenswaarden om outliers in de `bmi`-kolom te identificeren. Dit wordt gedaan met behulp van de interkwartielafstand (IQR). Meer uitleg hiervan waarom ik dit gedaan heb bij de `Data visualisation` Boxplot. 


```python
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

## Dataset Balanceren

De dataset wordt opgesplitst in twee klassen: 
- **Majority**: Personen zonder beroerte (stroke == 0)
- **Minority**: Personen met beroerte (stroke == 1)

Vervolgens wordt de meerderheidsklasse (zonder beroerte) *downsampled* naar de grootte van de minderheidsklasse (met beroerte), zodat beide klassen in balans zijn.


```python
# Splits de dataset op in majority (geen stroke) en minority (wel stroke)
df_majority = df[(df['stroke'] == 0) & (df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]
df_minority = df[(df['stroke'] == 1) & (df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]
# Neem een willekeurige sample van de meerderheidsklasse, gelijk aan de grootte van de minderheidsklasse
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # zonder vervanging
                                   n_samples=len(df_minority),  # aantal samples gelijk aan minority
                                   random_state=42)  # reproduceerbaarheid

# Combineer de minority-klasse en de downsampled majority-klasse
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Controleer de balans
print(df_balanced['stroke'].value_counts())
```

    stroke
    0    207
    1    207
    Name: count, dtype: int64
    


```python
df_balanced.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 414 entries, 3651 to 248
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   gender             414 non-null    object 
     1   age                414 non-null    float64
     2   hypertension       414 non-null    int64  
     3   heart_disease      414 non-null    int64  
     4   ever_married       414 non-null    object 
     5   work_type          414 non-null    object 
     6   Residence_type     414 non-null    object 
     7   avg_glucose_level  414 non-null    float64
     8   bmi                414 non-null    float64
     9   smoking_status     414 non-null    object 
     10  stroke             414 non-null    int64  
    dtypes: float64(3), int64(3), object(5)
    memory usage: 38.8+ KB
    

## Categorische Variabelen

In deze stap worden de categorische variabelen omgezet naar numerieke waarden, zodat ze door machine learning-modellen gebruikt kunnen worden. Hiervoor gebruiken we een eenvoudige mapping.

- **Geslacht** (`gender`): Omgezet naar 0 voor vrouw en 1 voor man.
- **Huwelijksstatus** (`ever_married`): Omgezet naar 0 voor 'No' en 1 voor 'Yes'.
- **Werktype** (`work_type`): Elke werkcategorie krijgt een numerieke waarde.
- **Type woning** (`Residence_type`): 'Rural' wordt 0 en 'Urban' wordt 1.
- **Rookstatus** (`smoking_status`): Rookcategorieën worden omgezet naar numerieke waarden, met een extra waarde voor onbekende status.


```python
# Maak een LabelEncoder-object
label_encoder = LabelEncoder()

# mapping gender
df_balanced['gender'] = df_balanced['gender'].map({'Female': 0, 'Male': 1})
# Mapping voor ever_married
df_balanced['ever_married'] = df_balanced['ever_married'].map({'No': 0, 'Yes': 1})

# Mapping voor work_type
df_balanced['work_type'] = df_balanced['work_type'].map({
    'children': 0,
    'Govt_job': 1,
    'Never_worked': 2,
    'Private': 3,
    'Self-employed': 4
})

# Mapping voor Residence_type
df_balanced['Residence_type'] = df_balanced['Residence_type'].map({'Rural': 0, 'Urban': 1})

# Mapping voor smoking_status
df_balanced['smoking_status'] = df_balanced['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
})
```


```python
df_balanced.sample(10)
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
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1783</th>
      <td>1</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65.67</td>
      <td>16.6</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>1</td>
      <td>64.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>210.00</td>
      <td>30.7</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1543</th>
      <td>1</td>
      <td>61.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>102.54</td>
      <td>40.5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3488</th>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>107.18</td>
      <td>27.6</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>70.30</td>
      <td>25.8</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1236</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>93.71</td>
      <td>31.2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>1</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>204.57</td>
      <td>34.4</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>0</td>
      <td>68.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>82.10</td>
      <td>27.1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106</th>
      <td>0</td>
      <td>50.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>73.18</td>
      <td>30.3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0</td>
      <td>78.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>106.74</td>
      <td>33.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Data visualisation

### Visualisatie van Gegevensverdeling

Ik heb een boxplot gegenereerd voor de `avg_glucose_level` en `bmi` kolommen om de verdeling van de data en eventuele outliers te visualiseren. Omdat er bij `BMI` een aantal outliers zaten heb ik de grenswaarden ingesteld door middel van IQR, om deze outliers eruit te filteren. De reden van deze outliers zal waarschijnlijk zijn door foutieve invulling bij het verzamelen van deze data.  


```python
# Boxplot voor avg_glucose_level en bmi
columns_to_check = ['avg_glucose_level', 'bmi']
df_balanced[columns_to_check].boxplot()
plt.title("Boxplot voor avg_glucose_level en bmi")
plt.ylabel("Waarden")
plt.xticks([1, 2], columns_to_check)
plt.show()
```


    
![png](output_15_0.png)
    


### Visualisatie van Genderverdeling

Ik heb een taartdiagram (pie chart) gemaakt om de verdeling van mannen en vrouwen in de dataset te visualiseren. Dit geeft inzicht in de verhouding van geslachten in de dataset. In mijn data komen meer vrouwen voor dat mannen.


```python
# Tel het aantal mannen en vrouwen
value = df_balanced['gender'].value_counts()

# Gebruik aangepaste labels
labels = ['Female', 'Male']

# Maak het taartdiagram
fig = go.Figure(data=[go.Pie(labels=labels, values=value, name="Gender", hole=0.4, textinfo="label+percent")])
fig.update_layout(title_text='Gender verdeling', title=dict(x=0.5))
fig.show()
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.32.0
* Copyright 2012-2024, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
