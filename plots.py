import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob

files = sorted(glob.glob("IMPUTED_DATA/All_Countries/*.csv"))

all_country_datasets = []

for file in files:
    df = pd.read_csv(file)
    all_country_datasets.append(df)

print(len(all_country_datasets))

# Add imputation number to each dataset
dfs = []

for i, data in enumerate(all_country_datasets, start=1):
    temp = data.copy()
    temp["imputation"] = i
    dfs.append(temp)

# Stack all 20 imputed datasets
stacked_df = pd.concat(dfs, ignore_index=True)

# Average each country-year across the 20 imputations
avg_df = (
    stacked_df
    .groupby(["country", "year"], as_index=False)
    .mean(numeric_only=True)
)

# Keep only needed variables
plot_df = avg_df[["country", "year", "gdp_growth", "palma_ratio"]].dropna()

# Fit simple regression line for the figure
X = sm.add_constant(plot_df["gdp_growth"])
y = plot_df["palma_ratio"]
model = sm.OLS(y, X).fit()

x_vals = np.linspace(plot_df["gdp_growth"].min(), plot_df["gdp_growth"].max(), 100)
y_vals = model.params["const"] + model.params["gdp_growth"] * x_vals

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(plot_df["gdp_growth"], plot_df["palma_ratio"], alpha=0.7)
plt.plot(x_vals, y_vals)

plt.xlabel("GDP Growth (%)")
plt.ylabel("Palma Ratio")
plt.title("Relationship Between GDP Growth and Palma Ratio")
plt.grid(True)
plt.show()