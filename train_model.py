import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
car = pd.read_csv("cleaned car.csv")

X = car[['name','company','year','kms_driven','fuel_type']]
y = car['Price']

ohe = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    [('ohe', ohe, ['name','company','fuel_type'])],
    remainder='passthrough'
)

model = LinearRegression()

pipe = Pipeline([
    ('step1', column_trans),
    ('step2', model)
])

pipe.fit(X,y)

pickle.dump(pipe, open('model.pkl','wb'))

print("Model trained successfully ✅")