import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
df = pd.read_csv('C:\\Users\\PIYUSH\\OneDrive\\Desktop\\new_ml\\credit_scoring - credit_scoring.csv')
# print(df)
print(df.info())
# transformer = ColumnTransformer(transformers=[
#     ('num',StandardScaler(),['age','income']),

num = df.select_dtypes(include=['int64','float64']).columns
cat = df.select_dtypes(include=['object']).columns
print(num)

transformer = ColumnTransformer(transformers=[
    ('num',StandardScaler(),num),
    ('cat',OneHotEncoder(),cat)
])
print(transformer.fit_transform(df))
