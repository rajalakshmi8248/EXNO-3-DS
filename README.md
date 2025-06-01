## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/73d4a022-ed4e-4d02-afbb-5e219e6f718e)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/40beddb9-00f1-4b9b-b0b6-b18463d7b1ea)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/e3410837-85d2-4389-b6d9-75c97579da3a)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/53c93641-1682-47bb-b684-0bb685dc6f8c)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/b40d6dc0-b28b-46df-a6f1-ce93ed4153b5)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/24427752-e370-484a-ae67-d474c463aef2)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/238b3c6a-27f2-4fca-995b-f9b801cfba95)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/a1c22c5f-4f93-4ac5-8b66-d98c068c1f25)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/78e25556-58f3-4d39-8b98-664882830377)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/6638fa9b-bb0f-4274-a74d-f8246a63bedf)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/b7eabde3-06f1-4d7d-80bb-fc8e59c3aeab)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/740229ec-b58c-4e8d-b614-564399f85a47)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/92cf3f8b-2116-4ed2-8417-44f954e51140)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ec205f6a-425e-4d55-bdf6-ba1dfacfef18)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5ca7597d-89fd-4e67-bdf7-a01dbc055cae)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8cc6fec9-1d7c-47c5-b4ab-f06ac4c3bb75)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/a52a34ee-f5e2-4816-a62f-9c0138c67e52)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/3f588a2c-167c-47e2-8271-62fec8891a39)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/2bb1b6cb-37dd-443d-af5e-cb521de814c1)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/66d10909-e4ee-4212-9600-afe7edac2ea9)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/user-attachments/assets/2cba81a0-862f-4005-88d9-e2487906967d)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6c6ec0a3-3e96-43dd-a6d7-8b1bbbaee818)

# RESULT:
Hence performing Feature Encoding and Transformation process is Successful.

       
