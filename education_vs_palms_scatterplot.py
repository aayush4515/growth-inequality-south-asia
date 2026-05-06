import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------------------------------
# 1. Folder path
# --------------------------------------------------

BASE_DIR = Path("IMPUTED_DATA")

countries = {
    "Bangladesh": BASE_DIR / "BGD",
    "India": BASE_DIR / "IND",
    "Nepal": BASE_DIR / "NPL",
    "Pakistan": BASE_DIR / "PAK",
    "Sri Lanka": BASE_DIR / "LKA"
}

# --------------------------------------------------
# 2. Function to aggregate one country's imputed files
# --------------------------------------------------

def aggregate_country_imputations(country_name, folder_path):
    csv_files = sorted(folder_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found for {country_name} in {folder_path}")

    all_dfs = []

    for i, file in enumerate(csv_files, start=1):
        df = pd.read_csv(file)

        required_cols = ["year", "palma_ratio", "education"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{file} is missing required column: {col}")

        temp = df[["year", "palma_ratio", "education"]].copy()
        temp["imputation"] = i
        temp["country"] = country_name

        all_dfs.append(temp)

    stacked = pd.concat(all_dfs, ignore_index=True)

    aggregated = (
        stacked
        .groupby(["country", "year"], as_index=False)
        .agg(
            avg_palma_ratio=("palma_ratio", "mean"),
            avg_education=("education", "mean")
        )
    )

    return aggregated

# --------------------------------------------------
# 3. Aggregate all countries
# --------------------------------------------------

country_results = []

for country_name, folder_path in countries.items():
    country_avg = aggregate_country_imputations(country_name, folder_path)
    country_results.append(country_avg)

pooled_aggregated = pd.concat(country_results, ignore_index=True)
pooled_aggregated = pooled_aggregated.sort_values(["country", "year"])

print(pooled_aggregated.head())
print(pooled_aggregated.shape)

# --------------------------------------------------
# 4. Prepare data for scatterplot
# --------------------------------------------------

plot_df = pooled_aggregated[["country", "year", "avg_education", "avg_palma_ratio"]].dropna()

# --------------------------------------------------
# 5. Compute LOWESS / LOESS smooth curve
# --------------------------------------------------
# frac controls how smooth the curve is.
# Higher frac = smoother curve
# Common values to try: 0.3, 0.4, 0.5, 0.6

lowess_result = lowess(
    endog=plot_df["avg_palma_ratio"],
    exog=plot_df["avg_education"],
    frac=0.4
)

# Separate x and y values for plotting
x_lowess = lowess_result[:, 0]
y_lowess = lowess_result[:, 1]

# --------------------------------------------------
# 6. Scatterplot: Education vs Palma Ratio
# --------------------------------------------------

plt.figure(figsize=(11, 6))

plt.scatter(
    plot_df["avg_education"],
    plot_df["avg_palma_ratio"],
    alpha=0.7
)

plt.plot(
    x_lowess,
    y_lowess,
    linewidth=2,
    label="LOWESS Curve"
)

plt.xlabel("Average Education")
plt.ylabel("Average Palma Ratio")
plt.title("Relationship Between Education and Palma Ratio\nAveraged Across 20 Imputed Datasets")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("education_vs_palma_ratio_lowess.png", dpi=300, bbox_inches="tight")
plt.show()