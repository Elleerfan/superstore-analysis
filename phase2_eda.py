import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv("data/superstore_clean.csv")
print(f"Rows :{df.shape[0]}")
print(f"Columns : {df.shape[1]}")

category= df.groupby("Category")[["Sales","Profit"]].sum().round(2)
print(category)

region = df.groupby("Region")[["Sales", "Profit"]].sum().round(2)
print(region)

monthly = df.groupby("Order Month")["Sales"].sum().round(2)
print(monthly)

top_products = df.groupby("Product Name")["Profit"].sum().round(2).sort_values(ascending=False).head(10)
print(top_products)

worst_products = df.groupby("Product Name")["Profit"].sum().round(2).sort_values(ascending=True).head(10)
print(worst_products)


monthly = df.groupby("Order Month")[["Sales", "Profit", "Discount"]].mean().round(2)
print("Monthly Averages:")
print(monthly)

yearly = df.groupby("Order Year")[["Sales", "Profit", "Discount"]].sum().round(2)
print("Yearly Totals:")
print(yearly)

state = df.groupby("State")[["Sales", "Discount"]].sum().round(2).sort_values("Sales", ascending=False).head(10)
print("Top 10 States by Sales:")
print(state)

monthly = df.groupby("Order Month")[["Sales", "Profit", "Discount"]].mean().round(2)

fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(monthly.index, monthly["Sales"], marker="o", color="steelblue", label="Avg Sales")
ax1.plot(monthly.index, monthly["Profit"], marker="o", color="green", label="Avg Profit")
ax1.set_xlabel("Month")
ax1.set_ylabel("Amount ($)")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

ax2 = ax1.twinx()
ax2.bar(monthly.index, monthly["Discount"], alpha=0.3, color="orange", label="Avg Discount")
ax2.set_ylabel("Discount")

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
plt.title("Monthly Avg Sales, Profit & Discount")
plt.tight_layout()
plt.savefig("monthly_trend_full.png")
plt.show()
print("Chart 8 saved!")