import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/superstore_clean.csv")

sns.set_theme(style="whitegrid")

print("Libraries loaded!")


category = df.groupby("Category")[["Sales", "Profit"]].sum().round(2)

category.plot(kind="bar", figsize=(8, 5), color=["steelblue", "orange"])
plt.title("Sales & Profit by Category")
plt.xlabel("Category")
plt.ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("category_chart.png")
plt.show()
print("Chart 1 saved!")

region = df.groupby("Region")["Sales"].sum().round(2)

plt.figure(figsize=(8, 5))
plt.pie(region, labels=region.index, autopct="%1.1f%%", startangle=90)
plt.title("Sales by Region")
plt.tight_layout()
plt.savefig("region_chart.png")
plt.show()
print("Chart 2 saved!")

monthly = df.groupby("Order Month")["Sales"].sum().round(2)



pivot = df.pivot_table(values="Profit", index="Region", columns="Category", aggfunc="sum").round(2)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn")
plt.title("Profit Heatmap by Region & Category")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Discount", y="Profit", alpha=0.5, hue="Category")
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("discount_vs_profit.png")
plt.show()
print(" Chart 5 saved!")

top_products = df.groupby("Product Name")["Profit"].sum().round(2).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette="Greens_r")
plt.title("Top 10 Most Profitable Products")
plt.xlabel("Total Profit ($)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("top_products.png")
plt.show()
print(" Chart 6 saved!")

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Segment", y="Sales", palette="Set2", estimator="mean")
plt.title("Average Sales by Segment")
plt.xlabel("Segment")
plt.ylabel("Average Sales ($)")
plt.tight_layout()
plt.savefig("segment_sales.png")
plt.show()
print("Chart 7 saved!")


shipmode = df.groupby("Ship Mode")["Sales"].sum().round(2)

plt.figure(figsize=(8, 8))
plt.pie(shipmode, labels=shipmode.index, autopct="%1.1f%%", startangle=90, colors=["steelblue","orange","green","#E98AD1"])
plt.title("Sales by Ship Mode")
plt.tight_layout()
plt.savefig("shipmode_pie.png")
plt.show()
print(" Ship Mode Chart saved!")




monthly = df.groupby(["Order Year", "Order Month"])[["Sales", "Profit", "Discount"]].sum().reset_index()
monthly["Date"] = pd.to_datetime(monthly["Order Year"].astype(str) + "-" + monthly["Order Month"].astype(str))
monthly = monthly.sort_values("Date")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Sales
ax1.plot(monthly["Date"], monthly["Sales"], color="steelblue", linewidth=2)
ax1.axhline(y=monthly["Sales"].mean(), color="red", linewidth=1.5, linestyle="--", label=f'Avg: ${monthly["Sales"].mean():.0f}')
ax1.set_title("Monthly Sales Over Years")
ax1.set_ylabel("Sales ($)")
ax1.legend()

# Profit
ax2.plot(monthly["Date"], monthly["Profit"], color="green", linewidth=2)
ax2.axhline(y=monthly["Profit"].mean(), color="red", linewidth=1.5, linestyle="--", label=f'Avg: ${monthly["Profit"].mean():.0f}')
ax2.set_title("Monthly Profit Over Years")
ax2.set_ylabel("Profit ($)")
ax2.legend()

# Discount
ax3.plot(monthly["Date"], monthly["Discount"], color="orange", linewidth=2)
ax3.axhline(y=monthly["Discount"].mean(), color="red", linewidth=1.5, linestyle="--", label=f'Avg: {monthly["Discount"].mean():.2f}')
ax3.set_title("Monthly Discount Over Years")
ax3.set_ylabel("Discount")
ax3.legend()

plt.tight_layout()
plt.savefig("monthly_trends.png")
plt.show()
print(" Chart 8 saved!")

yearly = df.groupby("Order Year")[["Sales", "Profit", "Discount"]].sum().round(2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

ax1.plot(yearly.index, yearly["Sales"], marker="o", color="steelblue", linewidth=2)
ax1.set_title("Yearly Total Sales")
ax1.set_ylabel("Sales ($)")
ax1.set_xticks(yearly.index)

ax2.plot(yearly.index, yearly["Profit"], marker="o", color="green", linewidth=2)
ax2.set_title("Yearly Total Profit")
ax2.set_ylabel("Profit ($)")
ax2.set_xticks(yearly.index)

ax3.plot(yearly.index, yearly["Discount"], marker="o", color="orange", linewidth=2)
ax3.set_title("Yearly Total Discount")
ax3.set_ylabel("Discount")
ax3.set_xticks(yearly.index)

plt.tight_layout()
plt.savefig("yearly_trends.png")
plt.show()
print("Chart 9 saved!")





top_states_sales = df.groupby('State')['Sales'].sum().nlargest(10).reset_index()
top_states_discount = df.groupby('State')['Discount'].sum().nlargest(10).reset_index()
top_states_profit = df.groupby('State')['Profit'].sum().nlargest(10).reset_index()
# Create the figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
# Plot for Sales
axes[0].barh(top_states_sales['State'], top_states_sales['Sales'], color= '#007d7c', edgecolor='black', linewidth=1)
axes[0].set_title('Top 10 States by Sales', fontsize=16)
axes[0].set_xlabel('Sales', fontsize=14)
axes[0].set_ylabel('State', fontsize=14)
axes[0].invert_yaxis()
axes[0].tick_params(axis='both', which='major', labelsize=12)
for i, v in enumerate(top_states_sales['Sales']):
    axes[0].text(v,i,f'{v:.2f}', va= 'center', ha= 'right', fontsize='10',fontweight='bold')
# Plot for Discount
axes[1].barh(top_states_discount['State'], top_states_discount['Discount'], color='#007d7c', edgecolor='black', linewidth=1)
axes[1].set_title('Top 10 States by Discount', fontsize=16)
axes[1].set_xlabel('Discount', fontsize=14)
axes[1].set_ylabel('State', fontsize=14)
axes[1].invert_yaxis()
axes[1].tick_params(axis='both', which='major', labelsize=12)   
for i, v in enumerate(top_states_discount['Discount']):
    axes[1].text(v,i,f'{v:.2f}',ha='right', fontsize='10', fontweight='bold')
# Plot for Profit
axes[2].barh(top_states_profit['State'], top_states_profit['Profit'], color='#007d7c', edgecolor='black', linewidth=1)
axes[2].set_title('Top 10 States by Profit', fontsize=16)
axes[2].set_xlabel('Profit', fontsize=14)
axes[2].set_ylabel('State', fontsize=14)
axes[2].invert_yaxis()
axes[2].tick_params(axis='both', which='major', labelsize=12)
for i, v, in enumerate(top_states_profit['Profit']):
    axes[2].text(v,i,f'{v:.2f}', ha='right', fontsize='10', fontweight='bold')
# Adjust the layout
plt.tight_layout()
# Display the plots 
plt.show()


top_cities_sales = df.groupby('City')['Sales'].sum().nlargest(10).reset_index()
top_cities_discount = df.groupby('City')['Discount'].sum().nlargest(10).reset_index()
top_cities_profit = df.groupby('City')['Profit'].sum().nlargest(10).reset_index()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for Sales
axes[0].barh(top_cities_sales['City'], top_cities_sales['Sales'], color='#007d7c', edgecolor='black', linewidth=1)
axes[0].set_title('Top 10 Cities by Sales', fontsize=16)
axes[0].set_xlabel('Sales', fontsize=14)
axes[0].set_ylabel('City', fontsize=14)
axes[0].invert_yaxis()
axes[0].tick_params(axis='both', which='major', labelsize=12)
for i, v in enumerate(top_cities_sales['Sales']):
    axes[0].text(v, i, f'{v:.2f}', va='center', ha='right', fontsize='10', fontweight='bold')

# Plot for Discount
axes[1].barh(top_cities_discount['City'], top_cities_discount['Discount'], color='#007d7c', edgecolor='black', linewidth=1)
axes[1].set_title('Top 10 Cities by Discount', fontsize=16)
axes[1].set_xlabel('Discount', fontsize=14)
axes[1].set_ylabel('City', fontsize=14)
axes[1].invert_yaxis()
axes[1].tick_params(axis='both', which='major', labelsize=12)
for i, v in enumerate(top_cities_discount['Discount']):
    axes[1].text(v, i, f'{v:.2f}', ha='right', fontsize='10', fontweight='bold')

# Plot for Profit
colors_profit = ["#e74c3c" if v < 0 else "#007d7c" for v in top_cities_profit['Profit']]
axes[2].barh(top_cities_profit['City'], top_cities_profit['Profit'], color=colors_profit, edgecolor='black', linewidth=1)
axes[2].set_title('Top 10 Cities by Profit', fontsize=16)
axes[2].set_xlabel('Profit', fontsize=14)
axes[2].set_ylabel('City', fontsize=14)
axes[2].invert_yaxis()
axes[2].tick_params(axis='both', which='major', labelsize=12)
for i, v in enumerate(top_cities_profit['Profit']):
    axes[2].text(v, i, f'{v:.2f}', ha='right', fontsize='10', fontweight='bold')

plt.tight_layout()
plt.savefig("top_cities.png")
plt.show()
print("Top 10 Cities chart saved!")