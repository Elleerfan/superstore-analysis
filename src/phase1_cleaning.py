import pandas as pd
import numpy as np

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin-1")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(df.head())
print(df.dtypes)
print(df.describe())
missing = df.isnull().sum()
print(missing)

df["Order Date"]=pd.to_datetime(df["Order Date"])
df["Ship Date"]=pd.to_datetime(df["Ship Date"])
print(df["Order Date"].dtype)
print(df["Ship Date"].dtype)

df["Days to Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin %"] = (df["Profit"]/df["Sales"]*100).round(2)
df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month
print(df[["Order Date", "Sales", "Profit", "Profit Margin %", "Days to Ship", "Order Year", "Order Month"]].head())
df.to_csv("data/superstore_clean.csv", index=False)
print(" Cleaned data saved!")
