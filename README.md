# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
from scipy import stats
import pandas as pd
df=pd.read_csv('/content/bmi.csv')
```
![Screenshot 2024-04-16 191022](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/dcec7afd-db09-422f-979a-47003b53dc5e)

![Screenshot 2024-04-16 191310](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/969e1758-2e87-422f-a7ec-c8bbbf1cc8ba)

![Screenshot 2024-04-16 191350](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/39eace21-2eaa-4967-a5c5-a7a4157a9a16)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Head','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 191521](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/88c1a4c7-9f30-4503-ad3e-1f214f560602)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 191608](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/6cde14b4-a726-4234-9459-8f737c521f20)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-04-16 191701](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/14517ad6-211e-42ef-8fe2-3113961ca3d4)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-04-16 191804](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/dfa28070-cc8f-4aa0-b522-6dcdbea2e79d)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-04-16 191817](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/ffe2d0dd-4d1d-49fe-b6f7-d5a65ef25b2b)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![Screenshot 2024-04-16 191949](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/40e5b49d-7456-4ce9-b275-82ff8088ef32)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
Selected Features:
Index(['Feature3'], dtype='object')
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head
```
![Screenshot 2024-04-16 192139](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/d41a80a8-3229-4581-b327-2413bc5d066b)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-04-16 192234](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/31d92cd1-b597-45fc-bc91-11e086d433e0)
```
chi2, p, _, _=chi2_contingency(contingency_table)
print("Chi-Square Statistic: {chi2}")
print(f"P-value:{p}")
```
![Screenshot 2024-04-16 192309](https://github.com/ThangaDeepika/EXNO-4-DS/assets/125663099/87e6605c-7a6f-4784-b8af-dc97e1b02b1e)

# RESULT:
       Feature Scaling and Feature Selection process was executed successfully.
