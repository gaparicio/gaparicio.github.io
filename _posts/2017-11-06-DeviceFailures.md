

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('failures.csv')
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>device</th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>S1F01085</td>
      <td>0</td>
      <td>215630672</td>
      <td>56</td>
      <td>0</td>
      <td>52</td>
      <td>6</td>
      <td>407438</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>S1F0166B</td>
      <td>0</td>
      <td>61370680</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>403174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>S1F01E6Y</td>
      <td>0</td>
      <td>173295968</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>237394</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>S1F01JE0</td>
      <td>0</td>
      <td>79694024</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>410186</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>S1F01R2B</td>
      <td>0</td>
      <td>135970480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>313173</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 124494 entries, 0 to 124493
    Data columns (total 12 columns):
    date          124494 non-null object
    device        124494 non-null object
    failure       124494 non-null int64
    attribute1    124494 non-null int64
    attribute2    124494 non-null int64
    attribute3    124494 non-null int64
    attribute4    124494 non-null int64
    attribute5    124494 non-null int64
    attribute6    124494 non-null int64
    attribute7    124494 non-null int64
    attribute8    124494 non-null int64
    attribute9    124494 non-null int64
    dtypes: int64(10), object(2)
    memory usage: 11.4+ MB
    


```python
#df.attribute7.value_counts()
```


```python
df.failure.value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xbff58d0>




![png](assets/2017-11-06-Device_Failures_4_1.png)



```python
df.failure.value_counts()
```




    0    124388
    1       106
    Name: failure, dtype: int64




```python
df.groupby('failure').mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
    <tr>
      <th>failure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.223827e+08</td>
      <td>156.118725</td>
      <td>9.945598</td>
      <td>1.696048</td>
      <td>14.221637</td>
      <td>260174.451056</td>
      <td>0.266682</td>
      <td>0.266682</td>
      <td>12.442462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.271755e+08</td>
      <td>4109.433962</td>
      <td>3.905660</td>
      <td>54.632075</td>
      <td>15.462264</td>
      <td>258303.481132</td>
      <td>30.622642</td>
      <td>30.622642</td>
      <td>23.084906</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(df.std(axis=0)).T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.029167</td>
      <td>7.045960e+07</td>
      <td>2179.65773</td>
      <td>185.747321</td>
      <td>22.908507</td>
      <td>15.943021</td>
      <td>99151.009852</td>
      <td>7.436924</td>
      <td>7.436924</td>
      <td>191.425623</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['attr1Bins'] = pd.qcut(df.attribute1, 10, labels=False)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>device</th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
      <th>attr1Bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>S1F01085</td>
      <td>0</td>
      <td>215630672</td>
      <td>56</td>
      <td>0</td>
      <td>52</td>
      <td>6</td>
      <td>407438</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01</td>
      <td>S1F0166B</td>
      <td>0</td>
      <td>61370680</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>403174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01</td>
      <td>S1F01E6Y</td>
      <td>0</td>
      <td>173295968</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>237394</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01</td>
      <td>S1F01JE0</td>
      <td>0</td>
      <td>79694024</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>410186</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01</td>
      <td>S1F01R2B</td>
      <td>0</td>
      <td>135970480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>313173</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(6,10))

ax1 = plt.subplot(331)
ax2 = plt.subplot(332)
ax3 = plt.subplot(333)
ax4 = plt.subplot(334)
ax5 = plt.subplot(335)
ax6 = plt.subplot(336)
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)

df.boxplot(column='attribute1',by= 'failure', ax=ax1)
df.boxplot(column='attribute2',by= 'failure', ax=ax2)
df.boxplot(column='attribute3',by= 'failure', ax=ax3)
df.boxplot(column='attribute4',by= 'failure', ax=ax4)
df.boxplot(column='attribute5',by= 'failure', ax=ax5)
df.boxplot(column='attribute6',by= 'failure', ax=ax6)
df.boxplot(column='attribute7',by= 'failure', ax=ax7)
df.boxplot(column='attribute8',by= 'failure', ax=ax8)
df.boxplot(column='attribute9',by= 'failure', ax=ax9)

plt.suptitle(' ')
plt.tight_layout()

```


![png](assets/2017-11-06-Device_Failures_9_0.png)



```python
df.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failure</th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
      <th>attr1Bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>failure</th>
      <td>1.000000</td>
      <td>0.001984</td>
      <td>0.052902</td>
      <td>-0.000948</td>
      <td>0.067398</td>
      <td>0.002270</td>
      <td>-0.000550</td>
      <td>0.119055</td>
      <td>0.119055</td>
      <td>0.001622</td>
      <td>0.001822</td>
    </tr>
    <tr>
      <th>attribute1</th>
      <td>0.001984</td>
      <td>1.000000</td>
      <td>-0.004248</td>
      <td>0.003702</td>
      <td>0.001837</td>
      <td>-0.003370</td>
      <td>-0.001516</td>
      <td>0.000151</td>
      <td>0.000151</td>
      <td>0.001122</td>
      <td>0.994968</td>
    </tr>
    <tr>
      <th>attribute2</th>
      <td>0.052902</td>
      <td>-0.004248</td>
      <td>1.000000</td>
      <td>-0.002617</td>
      <td>0.146593</td>
      <td>-0.013999</td>
      <td>-0.026350</td>
      <td>0.141367</td>
      <td>0.141367</td>
      <td>-0.002736</td>
      <td>-0.004191</td>
    </tr>
    <tr>
      <th>attribute3</th>
      <td>-0.000948</td>
      <td>0.003702</td>
      <td>-0.002617</td>
      <td>1.000000</td>
      <td>0.097452</td>
      <td>-0.006696</td>
      <td>0.009027</td>
      <td>-0.001884</td>
      <td>-0.001884</td>
      <td>0.532366</td>
      <td>0.003312</td>
    </tr>
    <tr>
      <th>attribute4</th>
      <td>0.067398</td>
      <td>0.001837</td>
      <td>0.146593</td>
      <td>0.097452</td>
      <td>1.000000</td>
      <td>-0.009773</td>
      <td>0.024870</td>
      <td>0.045631</td>
      <td>0.045631</td>
      <td>0.036069</td>
      <td>0.001700</td>
    </tr>
    <tr>
      <th>attribute5</th>
      <td>0.002270</td>
      <td>-0.003370</td>
      <td>-0.013999</td>
      <td>-0.006696</td>
      <td>-0.009773</td>
      <td>1.000000</td>
      <td>-0.017051</td>
      <td>-0.009384</td>
      <td>-0.009384</td>
      <td>0.005949</td>
      <td>-0.003391</td>
    </tr>
    <tr>
      <th>attribute6</th>
      <td>-0.000550</td>
      <td>-0.001516</td>
      <td>-0.026350</td>
      <td>0.009027</td>
      <td>0.024870</td>
      <td>-0.017051</td>
      <td>1.000000</td>
      <td>-0.012207</td>
      <td>-0.012207</td>
      <td>0.021152</td>
      <td>-0.001423</td>
    </tr>
    <tr>
      <th>attribute7</th>
      <td>0.119055</td>
      <td>0.000151</td>
      <td>0.141367</td>
      <td>-0.001884</td>
      <td>0.045631</td>
      <td>-0.009384</td>
      <td>-0.012207</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.006861</td>
      <td>-0.000204</td>
    </tr>
    <tr>
      <th>attribute8</th>
      <td>0.119055</td>
      <td>0.000151</td>
      <td>0.141367</td>
      <td>-0.001884</td>
      <td>0.045631</td>
      <td>-0.009384</td>
      <td>-0.012207</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.006861</td>
      <td>-0.000204</td>
    </tr>
    <tr>
      <th>attribute9</th>
      <td>0.001622</td>
      <td>0.001122</td>
      <td>-0.002736</td>
      <td>0.532366</td>
      <td>0.036069</td>
      <td>0.005949</td>
      <td>0.021152</td>
      <td>0.006861</td>
      <td>0.006861</td>
      <td>1.000000</td>
      <td>0.000653</td>
    </tr>
    <tr>
      <th>attr1Bins</th>
      <td>0.001822</td>
      <td>0.994968</td>
      <td>-0.004191</td>
      <td>0.003312</td>
      <td>0.001700</td>
      <td>-0.003391</td>
      <td>-0.001423</td>
      <td>-0.000204</td>
      <td>-0.000204</td>
      <td>0.000653</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc65cd30>




![png](assets/2017-11-06-Device_Failures_11_1.png)



```python
df.nunique()
```




    date             304
    device          1168
    failure            2
    attribute1    123878
    attribute2       558
    attribute3        47
    attribute4       115
    attribute5        60
    attribute6     44838
    attribute7        28
    attribute8        28
    attribute9        65
    attr1Bins         10
    dtype: int64




```python
print (df.duplicated().sum())
```

    0
    


```python
X = df.drop(['failure','device','date'], axis=1)
y = df.failure
```


```python
X.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attribute1</th>
      <th>attribute2</th>
      <th>attribute3</th>
      <th>attribute4</th>
      <th>attribute5</th>
      <th>attribute6</th>
      <th>attribute7</th>
      <th>attribute8</th>
      <th>attribute9</th>
      <th>attr1Bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215630672</td>
      <td>56</td>
      <td>0</td>
      <td>52</td>
      <td>6</td>
      <td>407438</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61370680</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>403174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>173295968</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>237394</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79694024</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>410186</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135970480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>313173</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: failure, dtype: int64




```python
from sklearn.cross_validation import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)
```

    C:\Users\gaparic\AppData\Local\Continuum\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    


```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=123, ratio=1.0)
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
X_train_smote.shape, y_train_smote.shape
```

    C:\Users\gaparic\AppData\Local\Continuum\Anaconda3\lib\site-packages\sklearn\utils\deprecation.py:75: DeprecationWarning: Function _ratio_float is deprecated; Use a float for 'ratio' is deprecated from version 0.2. The support will be removed in 0.4. Use a dict, str, or a callable instead.
      warnings.warn(msg, category=DeprecationWarning)
    




    ((174158, 10), (174158,))




```python
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(X_train_smote, y_train_smote)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print (clf_rf.score(X_test, y_test))
print (precision_score(y_test, clf_rf.predict(X_test)))
print (recall_score(y_test, clf_rf.predict(X_test)))
```

    0.998580952636
    0.157894736842
    0.075
    


```python
from sklearn.ensemble import GradientBoostingClassifier as GBC

gbc_model = GBC(n_estimators = 10)
gbc_model.fit(X_train_smote, y_train_smote)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print (gbc_model.score(X_test, y_test))
print (precision_score(y_test, gbc_model.predict(X_test)))
print (recall_score(y_test, gbc_model.predict(X_test)))
```

    0.916624273742
    0.00925925925926
    0.725
    


```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_sample(X_train, y_train)
X_train_rus.shape, y_train_rus.shape
```




    ((132, 10), (132,))




```python
from sklearn.ensemble import RandomForestClassifier
clf_rf1 = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf1.fit(X_train_rus, y_train_rus)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print (clf_rf1.score(X_test, y_test))
print (precision_score(y_test, clf_rf1.predict(X_test)))
print (recall_score(y_test, clf_rf1.predict(X_test)))
```

    0.840638303569
    0.00502344273275
    0.75
    


```python
from sklearn.ensemble import GradientBoostingClassifier as GBC

gbc_model1 = GBC(n_estimators = 10)
gbc_model1.fit(X_train_rus, y_train_rus)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print (gbc_model1.score(X_test, y_test))
print (precision_score(y_test, gbc_model1.predict(X_test)))
print (recall_score(y_test, gbc_model1.predict(X_test)))
```

    0.900131194945
    0.00826226012793
    0.775
    


```python
from imblearn.combine import SMOTEENN
sme = SMOTEENN(random_state=42)
X_train_sme, y_train_sme = sme.fit_sample(X_train, y_train)
X_train_sme.shape, y_train_sme.shape
```




    ((117991, 10), (117991,))




```python
from sklearn.ensemble import RandomForestClassifier
clf_rf2 = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf2.fit(X_train_sme, y_train_sme)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print (clf_rf2.score(X_test, y_test))
print (precision_score(y_test, clf_rf2.predict(X_test)))
print (recall_score(y_test, clf_rf2.predict(X_test)))
```

    0.997590296929
    0.0833333333333
    0.125
    


```python
from sklearn.ensemble import GradientBoostingClassifier as GBC

gbc_model2 = GBC(n_estimators = 10)
gbc_model2.fit(X_train_sme, y_train_sme)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


print (gbc_model2.score(X_test, y_test))
print (precision_score(y_test, gbc_model2.predict(X_test)))
print (recall_score(y_test, gbc_model2.predict(X_test)))
```

    0.916249431042
    0.0089058524173
    0.7
    


```python

```
