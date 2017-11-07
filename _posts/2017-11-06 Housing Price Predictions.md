

```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#import xgboost as xgb


from sklearn.preprocessing import StandardScaler
```


```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```


```python
train_Id = train['Id']
test_Id = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
```


```python
#train.iloc[1298]
train.drop(train[train.OverallQual > 5000].index, inplace=True)
train.drop(train[train.LotArea > 100000].index, inplace=True)
train.drop(train[train.GrLivArea > 4000].index, inplace=True)
train.drop(train[train.TotRmsAbvGrd > 13].index, inplace=True)
train.drop(train[train.GarageArea > 1200].index, inplace=True)
```


```python
y_train = train.SalePrice.values
df = pd.concat((train, test)).reset_index(drop=True)
df.drop(['SalePrice'], axis=1, inplace=True)
```


```python
sns.distplot(train['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc067be0>




![png](assets/2017-11-06 Housing Price Predictions_5_1.png)



```python
plt.figure(1)
f, axarr = plt.subplots(5, 2, figsize=(10, 20))
price = train.SalePrice.values
axarr[0, 0].scatter(train['YearBuilt'].values, price)
axarr[0, 0].set_title('YearBuilt')
axarr[0, 1].scatter(train.GrLivArea.values, price)
axarr[0, 1].set_title('GrLivArea')
axarr[1, 0].scatter(train.TotalBsmtSF.values, price)
axarr[1, 0].set_title('TotalBsmtSF')
axarr[1, 1].scatter(train['LotArea'].values, price)
axarr[1, 1].set_title('LotArea')
axarr[2, 0].scatter(train.OverallQual.values, price)
axarr[2, 0].set_title('OverallQual')
axarr[2, 1].scatter(train.TotRmsAbvGrd.values, price)
axarr[2, 1].set_title('TotRmsAbvGrd')
axarr[3, 0].scatter(train.YearRemodAdd.values, price)
axarr[3, 0].set_title('YearRemodAdd')
axarr[3, 1].scatter(train.FullBath.values, price)
axarr[3, 1].set_title('FullBath')
axarr[4, 0].scatter(train.TotRmsAbvGrd.values, price)
axarr[4, 0].set_title('TotRmsAbvGrd')
axarr[4, 1].scatter(train.GarageArea.values, price)
axarr[4, 1].set_title('GarageArea')

f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 18)
plt.tight_layout()
plt.show()

```


    <matplotlib.figure.Figure at 0xd51ecf8>



![png](assets/2017-11-06 Housing Price Predictions_6_1.png)



```python
#df.columns
```

## Data Analysis


```python
df.shape
```




    (2906, 79)




```python
corr = train.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, vmax=1, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd50b748>




![png](assets/2017-11-06 Housing Price Predictions_10_1.png)



```python
cor_dict = corr['SalePrice'].to_dict()
#del cor_dict['SalePrice']
print("List the numerical features decendingly by their correlation with Sale Price:\n")
for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))
```

    List the numerical features decendingly by their correlation with Sale Price:
    
    SalePrice: 	1.0
    OverallQual: 	0.8020448773679038
    GrLivArea: 	0.7226373192556508
    GarageCars: 	0.6559448993167509
    GarageArea: 	0.6503602030854637
    TotalBsmtSF: 	0.6424600311523415
    1stFlrSF: 	0.6210047418250213
    FullBath: 	0.5578250883517952
    TotRmsAbvGrd: 	0.5426895497232649
    YearBuilt: 	0.537769049973699
    YearRemodAdd: 	0.5227151070286912
    GarageYrBlt: 	0.5036384589885532
    MasVnrArea: 	0.47573072754265344
    Fireplaces: 	0.4623841527875624
    BsmtFinSF1: 	0.3901693843163795
    LotArea: 	0.3736187099489747
    LotFrontage: 	0.3524313102010717
    OpenPorchSF: 	0.332988305798493
    WoodDeckSF: 	0.32535770414743537
    2ndFlrSF: 	0.3030959221656794
    HalfBath: 	0.2887302052332916
    BsmtFullBath: 	0.2301720727745531
    BsmtUnfSF: 	0.2246306048340275
    BedroomAbvGr: 	0.16361269636462852
    KitchenAbvGr: 	-0.14035738196480563
    EnclosedPorch: 	-0.12864470092245392
    ScreenPorch: 	0.12080236886906373
    MSSubClass: 	-0.08717033951893287
    OverallCond: 	-0.08117012950383806
    MoSold: 	0.05792954890850577
    3SsnPorch: 	0.04844563367379602
    BsmtHalfBath: 	-0.039559885603267515
    PoolArea: 	0.03339925785942888
    LowQualFinSF: 	-0.027830515187285694
    YrSold: 	-0.02497857281157729
    MiscVal: 	-0.0212653131569129
    BsmtFinSF2: 	-0.016752892715550867
    

## Data Cleaning


```python

```


```python
#MSSubClass: The building class
#print(df.MSSubClass.isnull().sum())
#print(df.MSSubClass.value_counts())

df['MSSubClass'] = df['MSSubClass'].astype(str)
```


```python
#MSZoning: The general zoning classification
#print(df.MSZoning.isnull().sum())
#print(df.MSZoning.value_counts())

df['MSZoning'].fillna('NA', inplace=True)
```


```python
#LotFrontage: Linear feet of street connected to property
#print(df.LotFrontage.isnull().sum())
#print(df.LotFrontage.value_counts())
#print(df['LotFrontage'].mode())
#print(df['LotFrontage'].describe())
df['LotFrontage'].fillna((df['LotFrontage'].mean()), inplace=True)
#df['LotFrontage'] = StandardScaler().fit_transform(df['LotFrontage'].values.reshape(-1, 1))
```


```python
#LotArea: Lot size in square feet
#print(df.LotArea.isnull().sum())
#print(df.LotArea.value_counts())
#df['LotArea'].describe()
#df['LotArea'] = StandardScaler().fit_transform(df['LotArea'].values.reshape(-1, 1))
```


```python
df['LotArea'] = np.log(df['LotArea'])
```


```python
sns.distplot(df['LotArea'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd9ddac8>




![png](assets/2017-11-06 Housing Price Predictions_19_1.png)



```python
#Street: Type of road access
#print(df.Street.isnull().sum())
#print(df.Street.value_counts())

df['Street'] = df['Street'].apply(lambda x: x.replace(x, 'Yes') if x  == 'Pave' or x == 'Grvl' else x).apply(lambda x: x.replace(x, 'No') if x != 'Yes' else x )
pd.value_counts(df['Street'].values, sort =True)
```




    Yes    2906
    dtype: int64




```python
#Alley: Type of alley access
#print(df.Alley.isnull().sum())
#print(df.Alley.value_counts())

df['Alley'] = df['Alley'].fillna('No')
df['Alley'] = df['Alley'].apply(lambda x: x.replace(x, 'Yes') if x  == 'Pave' or x == 'Grvl' else x).apply(lambda x: x.replace(x, 'No') if x != 'Yes' else x )
pd.value_counts(df['Alley'].values, sort =True)
```




    No     2709
    Yes     197
    dtype: int64




```python
#LotShape: General shape of property
#print(df.LotShape.isnull().sum())
#print(df.LotShape.value_counts())
df['LotShape'] = df['LotShape'].replace({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1})
```


```python
#LandContour: Flatness of the property
#print(df.LandContour.isnull().sum())
#print(df.LandContour.value_counts())
```


```python
#Utilities: Type of utilities available
#print(df.Utilities.isnull().sum())
#print(df.Utilities.value_counts())

df['Utilities'] = df['Utilities'].fillna('AllPub')
pd.value_counts(df['Utilities'].values, sort =True)
```




    AllPub    2905
    NoSeWa       1
    dtype: int64




```python
#LotConfig: Lot configuration
#print(df.LotConfig.isnull().sum())
#print(df.LotConfig.value_counts())
```


```python
#LandSlope: Slope of property
#print(df.LandSlope.isnull().sum())
#print(df.LandSlope.value_counts())
```


```python
#Neighborhood: Physical locations within Ames city limits
#print(df.Neighborhood.isnull().sum())
#print(df.Neighborhood.value_counts())
```


```python
#Condition1: Proximity to main road or railroad
#print(df.Condition1.isnull().sum())
#print(df.Condition1.value_counts())
```


```python
#Condition2: Proximity to main road or railroad (if a second is present)
#print(df.Condition2.isnull().sum())
#print(df.Condition2.value_counts())
```


```python
#BldgType: Type of dwelling
#print(df.BldgType.isnull().sum())
#print(df.BldgType.value_counts())
```


```python
#HouseStyle: Style of dwelling
#print(df.HouseStyle.isnull().sum())
#print(df.HouseStyle.value_counts())
```


```python
#OverallQual: Overall material and finish quality
#print(df.OverallQual.isnull().sum())
#print(df.OverallQual.value_counts())
df['OverallQual'] = df['OverallQual'].astype(str)
```


```python
#OverallCond: Overall condition rating
#print(df.OverallCond.isnull().sum())
#print(df.OverallCond.value_counts())
df['OverallCond'] = df['OverallCond'].astype(str)
```


```python
#YearBuilt: Original construction date
#print(df.YearBuilt.isnull().sum())
#print(df.YearBuilt.value_counts())
#df.YearBuilt.describe()
#df['YearBuilt'] = StandardScaler().fit_transform(df['YearBuilt'].values.reshape(-1, 1))
```


```python
#YearRemodAdd: Remodel date
#print(df.YearRemodAdd.isnull().sum())
#print(df.YearRemodAdd.value_counts())
#df.YearRemodAdd.describe()
#df['YearRemodAdd'] = StandardScaler().fit_transform(df['YearRemodAdd'].values.reshape(-1, 1))
```


```python
df['YearRemodAdd'] = np.log(df['YearRemodAdd'])
```


```python
sns.distplot(df['YearRemodAdd'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe4e4a58>




![png](assets/2017-11-06 Housing Price Predictions_37_1.png)



```python
#RoofStyle: Type of roof
#print(df.RoofStyle.isnull().sum())
#print(df.RoofStyle.value_counts())
```


```python
#RoofMatl: Roof material
#print(df.RoofMatl.isnull().sum())
#print(df.RoofMatl.value_counts())
```


```python
#Exterior1st: Exterior covering on house
#print(df.Exterior1st.isnull().sum())
#print(df.Exterior1st.value_counts())
df['Exterior1st'].fillna('NA', inplace=True)
```


```python
#Exterior2nd: Exterior covering on house (if more than one material)
#print(df.Exterior2nd.isnull().sum())
#print(df.Exterior2nd.value_counts())
df['Exterior2nd'].fillna('NA', inplace=True)
```


```python
#MasVnrType: Masonry veneer type
#print(df.MasVnrType.isnull().sum())
#print(df.MasVnrType.value_counts())
df['MasVnrType'].fillna('None', inplace=True)
```


```python
#MasVnrArea: Masonry veneer area in square feet
#print(df.MasVnrArea.isnull().sum())
#print(df.MasVnrArea.value_counts())
df['MasVnrArea'].fillna(0, inplace=True)
#df['MasVnrArea'] = StandardScaler().fit_transform(df['MasVnrArea'].values.reshape(-1, 1))
```


```python
#ExterQual: Exterior material quality
#print(df.ExterQual.isnull().sum())
#print(df.ExterQual.value_counts())

eq={'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}
df['ExterQual'] = df['ExterQual'].replace(eq,regex=True)
```


```python
#ExterCond: Present condition of the material on the exterior
#print(df.ExterCond.isnull().sum())
#print(df.ExterCond.value_counts())

ec={'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}
df['ExterCond'] = df['ExterCond'].replace(ec,regex=True)
```


```python
#Foundation: Type of foundation
#print(df.Foundation.isnull().sum())
#print(df.Foundation.value_counts())
```


```python
#BsmtQual: Height of the basement
#print(df.BsmtQual.isnull().sum())
#print(df.BsmtQual.value_counts())
#bq={'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}
#df['BsmtQual'] = df['BsmtQual'].replace(bq,regex=True)
```


```python
#BsmtCond: General condition of the basement
#BsmtExposure: Walkout or garden level basement walls
#BsmtFinType1: Quality of basement finished area
#BsmtFinSF1: Type 1 finished square feet
#BsmtFinType2: Quality of second finished area (if present)
#BsmtFinSF2: Type 2 finished square feet
#print(df.BsmtCond.isnull().sum())
#print(df.BsmtCond.value_counts())

#translate to hasBasement yes, no
df['BsmtCond'] = df['BsmtCond'].fillna('No')
df['BsmtCond'] = df['BsmtCond'].apply(lambda x: x.replace(x, 'Yes') if x  == 'TA' or x == 'Gd' or x == 'Ex'  or x == 'Fa'  or x == 'Po' else x).apply(lambda x: x.replace(x, 'No') if x != 'Yes' else x )
pd.value_counts(df['BsmtCond'].values, sort =True)
```




    Yes    2824
    No       82
    dtype: int64




```python
#BsmtUnfSF: Unfinished square feet of basement area
#print(df.BsmtUnfSF.isnull().sum())
#print(df.BsmtUnfSF.value_counts())
#print(df['BsmtUnfSF'].describe())
df['BsmtUnfSF'].fillna((df['BsmtUnfSF'].mean()), inplace=True)
#df['BsmtUnfSF'] = StandardScaler().fit_transform(df['BsmtUnfSF'].values.reshape(-1, 1))
```


```python
#TotalBsmtSF: Total square feet of basement area
#print(df.TotalBsmtSF.isnull().sum())
#print(df.TotalBsmtSF.value_counts())
#print(df['TotalBsmtSF'].describe())
df['TotalBsmtSF'].fillna((df['TotalBsmtSF'].mean()), inplace=True)
#df['TotalBsmtSF'] = StandardScaler().fit_transform(df['TotalBsmtSF'].values.reshape(-1, 1))
```


```python
#df['TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
```


```python
sns.distplot(df['TotalBsmtSF'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe54eda0>




![png](assets/2017-11-06 Housing Price Predictions_52_1.png)



```python
#Heating: Type of heating
#HeatingQC: Heating quality and condition
#print(df.Heating.isnull().sum())
#print(df.Heating.value_counts())
```


```python
#CentralAir: Central air conditioning
#print(df.CentralAir.isnull().sum())
#print(df.CentralAir.value_counts())
```


```python
#Electrical: Electrical system
#print(df.Electrical.isnull().sum())
#print(df.Electrical.value_counts())

df['Electrical'].fillna('SBrkr', inplace=True)
```


```python
#1stFlrSF: First Floor square feet
#print(df['1stFlrSF'].isnull().sum())
#print(df['1stFlrSF'].value_counts())
#df['1stFlrSF'].describe()
#df['1stFlrSF'] = StandardScaler().fit_transform(df['1stFlrSF'].values.reshape(-1, 1))
```


```python
#2ndFlrSF: Second floor square feet
#print(df['2ndFlrSF'].isnull().sum())
#print(df['2ndFlrSF'].value_counts())
#df['2ndFlrSF'].describe()
#df['2ndFlrSF'] = StandardScaler().fit_transform(df['2ndFlrSF'].values.reshape(-1, 1))
```


```python
#LowQualFinSF: Low quality finished square feet (all floors)
#print(df.LowQualFinSF.isnull().sum())
#print(df.LowQualFinSF.value_counts())
#df['LowQualFinSF'].describe()
#df['LowQualFinSF'] = StandardScaler().fit_transform(df['LowQualFinSF'].values.reshape(-1, 1))
```


```python
#GrLivArea: Above grade (ground) living area square feet
#print(df.GrLivArea.isnull().sum())
#print(df.GrLivArea.value_counts())
#df['GrLivArea'].describe()
#df['GrLivArea'] = StandardScaler().fit_transform(df['GrLivArea'].values.reshape(-1, 1))
```


```python
df['GrLivArea'] = np.log(df['GrLivArea'])
```


```python
sns.distplot(df['GrLivArea'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe82cdd8>




![png](assets/2017-11-06 Housing Price Predictions_61_1.png)



```python
#BsmtFullBath: Basement full bathrooms
#print(df.BsmtFullBath.isnull().sum())
#print(df.BsmtFullBath.value_counts())

df['BsmtFullBath'].fillna((df['BsmtFullBath'].mean()), inplace=True)
```


```python
#BsmtHalfBath: Basement half bathrooms
#print(df.BsmtHalfBath.isnull().sum())
#print(df.BsmtHalfBath.value_counts())
df['BsmtHalfBath'].fillna((df['BsmtHalfBath'].mean()), inplace=True)
```


```python
#FullBath: Full bathrooms above grade
#print(df.FullBath.isnull().sum())
#print(df.FullBath.value_counts())
#df['FullBath'] = StandardScaler().fit_transform(df['FullBath'].values.reshape(-1, 1))
```


```python
#df['FullBath'] = np.log(df['FullBath'])
```


```python
sns.distplot(df['FullBath'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe948908>




![png](assets/2017-11-06 Housing Price Predictions_66_1.png)



```python
#HalfBath: Half baths above grade
#print(df.HalfBath.isnull().sum())
#print(df.HalfBath.value_counts())
```


```python
#BedroomAbvGr: Number of bedrooms above basement level
#print(df.BedroomAbvGr.isnull().sum())
#print(df.BedroomAbvGr.value_counts())
```


```python
#KitchenAbvGr: Number of kitchens
#print(df.KitchenAbvGr.isnull().sum())
#print(df.KitchenAbvGr.value_counts())
```


```python
#KitchenQual: Kitchen quality
#print(df.KitchenQual.isnull().sum())
#print(df.KitchenQual.value_counts())

kq={'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}
df['KitchenQual'] = df['KitchenQual'].replace(kq,regex=True)
df['KitchenQual'].fillna((df['KitchenQual'].mean()), inplace=True)
```


```python
#TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#print(df.TotRmsAbvGrd.isnull().sum())
#print(df.TotRmsAbvGrd.value_counts())
#df['TotRmsAbvGrd'] = StandardScaler().fit_transform(df['TotRmsAbvGrd'].values.reshape(-1, 1))
```


```python
df['TotRmsAbvGrd'] = np.log(df['TotRmsAbvGrd'])
```


```python
sns.distplot(df['TotRmsAbvGrd'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xea946d8>




![png](assets/2017-11-06 Housing Price Predictions_73_1.png)



```python
#Functional: Home functionality rating
#print(df.Functional.isnull().sum())
#print(df.Functional.value_counts())
df['Functional'].fillna('Typ', inplace=True)
```


```python
#Fireplaces: Number of fireplaces
#FireplaceQu: Fireplace quality
#print(df.Fireplaces.isnull().sum())
#print(df.Fireplaces.value_counts())
#df['Fireplaces'] = StandardScaler().fit_transform(df['Fireplaces'].values.reshape(-1, 1))
```


```python
#GarageType: Garage location
#print(df.GarageType.isnull().sum())
#print(df.GarageType.value_counts())
```


```python
#GarageYrBlt: Year garage was built
#print(df.GarageYrBlt.isnull().sum())
#print(df.GarageYrBlt.value_counts())
df['GarageYrBlt'].fillna((df['GarageYrBlt'].mean()), inplace=True)
```


```python
#GarageFinish: Interior finish of the garage
#print(df.GarageFinish.isnull().sum())
#print(df.GarageFinish.value_counts())
```


```python
#GarageCars: Size of garage in car capacity
#print(df.GarageCars.isnull().sum())
#print(df.GarageCars.value_counts())
df['GarageCars'].fillna((df['GarageCars'].mean()), inplace=True)
#df['GarageCars'] = StandardScaler().fit_transform(df['GarageCars'].values.reshape(-1, 1))
```


```python
#df['GarageCars'] = np.log(df['GarageCars'])
```


```python
sns.distplot(df['GarageCars'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xeaaf1d0>




![png](assets/2017-11-06 Housing Price Predictions_81_1.png)



```python
#GarageArea: Size of garage in square feet
#print(df.GarageArea.isnull().sum())
#print(df.GarageArea.value_counts())
df['GarageArea'].fillna((df['GarageArea'].mean()), inplace=True)
#df['GarageArea'] = StandardScaler().fit_transform(df['GarageArea'].values.reshape(-1, 1))
```


```python
#df['GarageArea'] = np.log(df['GarageArea'])
```


```python
sns.distplot(df['GarageArea'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe9bb630>




![png](assets/2017-11-06 Housing Price Predictions_84_1.png)



```python
#GarageQual: Garage quality
#print(df.GarageQual.isnull().sum())
#print(df.GarageQual.value_counts())
```


```python
#GarageCond: Garage condition
#print(df.GarageCond.isnull().sum())
#print(df.GarageCond.value_counts())
```


```python
#PavedDrive: Paved driveway
#print(df.PavedDrive.isnull().sum())
#print(df.PavedDrive.value_counts())
```


```python
#WoodDeckSF: Wood deck area in square feet
#print(df.WoodDeckSF.isnull().sum())
#print(df.WoodDeckSF.value_counts())
#df['WoodDeckSF'].describe()
#df['WoodDeckSF'] = StandardScaler().fit_transform(df['WoodDeckSF'].values.reshape(-1, 1))
```


```python
#OpenPorchSF: Open porch area in square feet
#print(df.OpenPorchSF.isnull().sum())
#print(df.OpenPorchSF.value_counts())
#df['OpenPorchSF'].describe()
#df['OpenPorchSF'] = StandardScaler().fit_transform(df['OpenPorchSF'].values.reshape(-1, 1))
```


```python
#EnclosedPorch: Enclosed porch area in square feet
#print(df.EnclosedPorch.isnull().sum())
#print(df.EnclosedPorch.value_counts())
#df['EnclosedPorch'].describe()
#df['EnclosedPorch'] = StandardScaler().fit_transform(df['EnclosedPorch'].values.reshape(-1, 1))
```


```python
#3SsnPorch: Three season porch area in square feet
#print(df['3SsnPorch'].isnull().sum())
#print(df['3SsnPorch'].value_counts())
#df['3SsnPorch'].describe()
#df['3SsnPorch'] = StandardScaler().fit_transform(df['3SsnPorch'].values.reshape(-1, 1))
```


```python
#ScreenPorch: Screen porch area in square feet
#print(df.ScreenPorch.isnull().sum())
#print(df.ScreenPorch.value_counts())
#df['ScreenPorch'].describe()
#df['ScreenPorch'] = StandardScaler().fit_transform(df['ScreenPorch'].values.reshape(-1, 1))
```


```python
#PoolArea: Pool area in square feet
#print(df.PoolArea.isnull().sum())
#print(df.PoolArea.value_counts())
#df['PoolArea'] = StandardScaler().fit_transform(df['PoolArea'].values.reshape(-1, 1))
```


```python
#PoolQC: Pool quality
#print(df.PoolQC.isnull().sum())
#print(df.PoolQC.value_counts())

df['PoolQC']= df['PoolQC'].fillna('No')
df['PoolQC'] = df['PoolQC'].apply(lambda x: x.replace(x, 'Yes') if x == 'Ex' or  x == 'Gd' or x == 'Fa' else x)
pd.value_counts(df['PoolQC'].values, sort=True)
```




    No     2898
    Yes       8
    dtype: int64




```python
#Fence: Fence quality
#print(df.Fence.isnull().sum())
#print(df.Fence.value_counts())

df['Fence'] = df['Fence'].fillna('No')
df['Fence'] = df['Fence'].apply(lambda x: x.replace(x, 'Yes') if x  == 'MnPrv' or x == 'GdPrv' or x =='GdWo' or x == 'MnWw' else x).apply(lambda x: x.replace(x, 'No') if x != 'Yes' else x )
pd.value_counts(df['Fence'].values, sort =True)
```




    No     2336
    Yes     570
    dtype: int64




```python
#MiscFeature: Miscellaneous feature not covered in other categories
#print(df.MiscFeature.isnull().sum())
#print(df.MiscFeature.value_counts())
```


```python
#MiscVal: $Value of miscellaneous feature
#print(df.MiscVal.isnull().sum())
#print(df.MiscVal.value_counts())
```


```python
#MoSold: Month Sold
#print(df.MoSold.isnull().sum())
#print(df.MoSold.value_counts())
df['MoSold'] = df['MoSold'].astype(str)
```


```python
#YrSold: Year Sold
#print(df.YrSold.isnull().sum())
#print(df.YrSold.value_counts())
#df['YrSold'] = df['YrSold'].astype(str)
```


```python
#SaleType: Type of sale
#print(df.SaleType.isnull().sum())
#print(df.SaleType.value_counts())

df['SaleType'] = df['SaleType'].fillna('WD')
```


```python
#SaleCondition: Condition of sale
#print(df.SaleCondition.isnull().sum())
#print(df.SaleCondition.value_counts())
```

## Feature Engineering


```python
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
#df['TotalSF'] = StandardScaler().fit_transform(df['TotalSF'].values.reshape(-1, 1))
```


```python
df['YrSold'] = 2017 - df['YrSold']
```

## Data Prep


```python
categorical = pd.DataFrame(df[[

#'MSSubClass',
#'MSZoning',
#'Street',
#'Alley',
#'LotShape',
#'LandContour',
#'Utilities',
#'LotConfig',
#'LandSlope',
#'Neighborhood',
#'Condition1',
#'Condition2',
#'BldgType',
#'HouseStyle',
'OverallQual',#
#'OverallCond',
#'RoofStyle',
#'RoofMatl',
#'Exterior1st',
#'Exterior2nd',
#'MasVnrType',
#'Foundation',
#'BsmtCond',
#'BsmtUnfSF',
#'Heating',
#'CentralAir',
#'Electrical',
#'Functional',
#'PavedDrive',
#'PoolQC',
#'Fence',
#'MoSold',
#'YrSold',
#'SaleType',
#'SaleCondition',
]])

print(categorical.shape)
categorical.head()
```

    (2906, 1)
    




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
      <th>OverallQual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
numerical = pd.DataFrame(df[[
#'LotFrontage',    
'LotArea',
#'YearBuilt', #
'YearRemodAdd',#
#'MasVnrArea',
#'ExterQual',
#'ExterCond',
'TotalBsmtSF',#
#'1stFlrSF',#
#'2ndFlrSF',
#'LowQualFinSF',
'GrLivArea',#
#'BsmtFullBath',
'FullBath',#
#'HalfBath',
#'BedroomAbvGr',
#'KitchenAbvGr',
#'KitchenQual',
'TotRmsAbvGrd',#
#'GarageYrBlt',
#'Fireplaces',
'GarageCars',#
'GarageArea',#    
#'WoodDeckSF',
#'OpenPorchSF',
#'EnclosedPorch',
#'3SsnPorch',
#'ScreenPorch',
#'PoolArea',
#'TotalSF',
'YrSold',
'LotShape',
]])

print(numerical.shape)
numerical.head()
```

    (2906, 10)
    




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
      <th>LotArea</th>
      <th>YearRemodAdd</th>
      <th>TotalBsmtSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>YrSold</th>
      <th>LotShape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.041922</td>
      <td>7.602401</td>
      <td>856.0</td>
      <td>7.444249</td>
      <td>2</td>
      <td>2.079442</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.169518</td>
      <td>7.588830</td>
      <td>1262.0</td>
      <td>7.140453</td>
      <td>2</td>
      <td>1.791759</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.328123</td>
      <td>7.601902</td>
      <td>920.0</td>
      <td>7.487734</td>
      <td>2</td>
      <td>1.791759</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.164296</td>
      <td>7.585789</td>
      <td>756.0</td>
      <td>7.448334</td>
      <td>1</td>
      <td>1.945910</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>11</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.565214</td>
      <td>7.600902</td>
      <td>1145.0</td>
      <td>7.695303</td>
      <td>2</td>
      <td>2.197225</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>9</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical = pd.get_dummies(categorical,drop_first=True)
categorical.head()
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
      <th>OverallQual_10</th>
      <th>OverallQual_2</th>
      <th>OverallQual_3</th>
      <th>OverallQual_4</th>
      <th>OverallQual_5</th>
      <th>OverallQual_6</th>
      <th>OverallQual_7</th>
      <th>OverallQual_8</th>
      <th>OverallQual_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
from scipy.stats import norm, skew

skewed_feats = numerical.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew': skewed_feats})
skewness
'''
```




    "\nfrom scipy.stats import norm, skew\n\nskewed_feats = numerical.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)\n\nskewness = pd.DataFrame({'Skew': skewed_feats})\nskewness\n"




```python
'''
from scipy.special import boxcox1p

skewness = skewness[abs(skewness) > 0.75]
print ("There are", skewness.shape[0], "skewed numerical features to Box Cox transform")

skewed_features = skewness.index
lmbda = 0.15
for feat in skewed_features:
    numerical[feat] = boxcox1p(df[feat], lmbda)
    numerical[feat] += 1
'''
```




    '\nfrom scipy.special import boxcox1p\n\nskewness = skewness[abs(skewness) > 0.75]\nprint ("There are", skewness.shape[0], "skewed numerical features to Box Cox transform")\n\nskewed_features = skewness.index\nlmbda = 0.15\nfor feat in skewed_features:\n    numerical[feat] = boxcox1p(df[feat], lmbda)\n    numerical[feat] += 1\n'




```python
data = pd.concat([categorical,numerical],axis=1)
print (data.shape)
data.head()
```

    (2906, 19)
    




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
      <th>OverallQual_10</th>
      <th>OverallQual_2</th>
      <th>OverallQual_3</th>
      <th>OverallQual_4</th>
      <th>OverallQual_5</th>
      <th>OverallQual_6</th>
      <th>OverallQual_7</th>
      <th>OverallQual_8</th>
      <th>OverallQual_9</th>
      <th>LotArea</th>
      <th>YearRemodAdd</th>
      <th>TotalBsmtSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>YrSold</th>
      <th>LotShape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9.041922</td>
      <td>7.602401</td>
      <td>856.0</td>
      <td>7.444249</td>
      <td>2</td>
      <td>2.079442</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9.169518</td>
      <td>7.588830</td>
      <td>1262.0</td>
      <td>7.140453</td>
      <td>2</td>
      <td>1.791759</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9.328123</td>
      <td>7.601902</td>
      <td>920.0</td>
      <td>7.487734</td>
      <td>2</td>
      <td>1.791759</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9.164296</td>
      <td>7.585789</td>
      <td>756.0</td>
      <td>7.448334</td>
      <td>1</td>
      <td>1.945910</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>11</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9.565214</td>
      <td>7.600902</td>
      <td>1145.0</td>
      <td>7.695303</td>
      <td>2</td>
      <td>2.197225</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>9</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = data[:len(train)]
test_df = data[len(train):]
```

## Feature Importance Investigation


```python
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


rfc = RandomForestRegressor()
rfc_model = rfc.fit(train_df.values, y_train)
importances = pd.DataFrame({'feature':train_df.columns,'importance':np.round(rfc.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
 
print (importances[:20])
importances[:20].plot.bar()
```

                    importance
    feature                   
    GarageCars           0.311
    GrLivArea            0.285
    TotalBsmtSF          0.153
    YearRemodAdd         0.073
    LotArea              0.042
    GarageArea           0.037
    OverallQual_8        0.033
    TotRmsAbvGrd         0.017
    OverallQual_7        0.010
    YrSold               0.006
    OverallQual_5        0.006
    OverallQual_9        0.006
    OverallQual_6        0.005
    LotShape             0.004
    FullBath             0.004
    OverallQual_4        0.003
    OverallQual_3        0.002
    OverallQual_10       0.002
    OverallQual_2        0.000
    




    <matplotlib.axes._subplots.AxesSubplot at 0x10536f98>




![png](assets/2017-11-06 Housing Price Predictions_114_2.png)



```python
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor


gbc = GradientBoostingRegressor(n_estimators=10)
model_gbc = gbc.fit(train_df.values, y_train)
gbc_importances = pd.DataFrame({'feature':train_df.columns,'importance':np.round(gbc.feature_importances_,3)})
gbc_importances = gbc_importances.sort_values('importance',ascending=False).set_index('feature')
 
print (gbc_importances[:20])
gbc_importances[:20].plot.bar()
```

                    importance
    feature                   
    GrLivArea            0.348
    GarageCars           0.252
    TotalBsmtSF          0.229
    YearRemodAdd         0.090
    GarageArea           0.031
    FullBath             0.021
    OverallQual_8        0.016
    OverallQual_5        0.007
    OverallQual_9        0.005
    OverallQual_7        0.001
    OverallQual_10       0.000
    YrSold               0.000
    TotRmsAbvGrd         0.000
    LotArea              0.000
    OverallQual_2        0.000
    OverallQual_6        0.000
    OverallQual_4        0.000
    OverallQual_3        0.000
    LotShape             0.000
    




    <matplotlib.axes._subplots.AxesSubplot at 0x105f8d68>




![png](assets/2017-11-06 Housing Price Predictions_115_2.png)



```python
from sklearn.cross_validation import train_test_split

y = train.SalePrice.values
X = train_df

#print(X.shape)
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
```

    C:\Users\gaparic\AppData\Local\Continuum\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

## Model


```python
from sklearn.metrics import mean_squared_error,r2_score

rfr = RandomForestRegressor(n_estimators=20)
model_rfr = rfr.fit(X_train,y_train)

rfr_pred = model_rfr.predict(X_test)

print ('mean_squared_error: ',mean_squared_error(y_test,rfr_pred)**0.5)
print ('r2_score: ',r2_score(y_test,rfr_pred))
```

    mean_squared_error:  26385.7680309
    r2_score:  0.86635869137
    


```python
from sklearn.metrics import mean_squared_error,r2_score

gbr = GradientBoostingRegressor(n_estimators=20)
model_gbr = gbr.fit(X_train,y_train)

gbr_pred = model_gbr.predict(X_test)

print ('mean_squared_error: ',mean_squared_error(y_test,gbr_pred)**0.5)
print ('r2_score: ',r2_score(y_test,gbr_pred))
```

    mean_squared_error:  28786.3353885
    r2_score:  0.840935230077
    


```python
from sklearn import linear_model
clf = linear_model.Lasso(max_iter=1e6, alpha=5e-4)
model_clf = clf.fit(X_train,y_train)

clf_pred = model_clf.predict(X_test)

print ('mean_squared_error: ',mean_squared_error(y_test,clf_pred)**0.5)
print ('r2_score: ',r2_score(y_test,clf_pred))

```

    mean_squared_error:  25971.993107
    r2_score:  0.870517284909
    


```python
#import xgboost as xgb

#model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, learning_rate=0.05, max_depth=6, min_child_weight=1.5, n_estimators=7200, reg_alpha=0.9, reg_lambda=0.6, subsample=0.2,seed=42, silent=1)

#model_xgb.fit(X_train,y_train)

#xgb_pred = model_xgb.predict(X_test)


#print ('mean_squared_error: ',mean_squared_error(y_test,xgb_pred)**0.5)
#print ('r2_score: ',r2_score(y_test,xgb_pred))
```

## Prediction


```python
clf_test_pred = model_clf.predict(test_df)
```


```python
df2 = pd.DataFrame(test_Id)
df3 = pd.DataFrame(clf_test_pred)
df4 = pd.merge(df2,df3,left_index=True,right_index=True)
df4 = df4.rename(columns={0: 'SalePrice'})
print(df4.shape)
df4.head()
```


```python
#df4.to_csv('171024_Housing Prices_export_07.csv',index=False)
```


```python

```
