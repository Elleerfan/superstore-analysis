import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("data/superstore_clean.csv")


Q1 = df["Profit"].quantile(0.25)
Q3 = df["Profit"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["Profit"] >= Q1 - 1.5 * IQR) & (df["Profit"] <= Q3 + 1.5 * IQR)]

print(f"Rows after removing outliers: {df.shape[0]}")


features = pd.get_dummies(df[["Quantity", "Discount", "Category", "Region", "Segment", "Ship Mode"]])
target = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : ${mae:.2f}")
print(f"R2 Score            : {r2:.4f}")