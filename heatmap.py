import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
# 2. Variables to include in the heatmap
# --------------------------------------------------

variables = [
    "palma_ratio",
    "gdp_growth",
    "education",
    "fdi",
    "log_gdp_pc",
    "industry",
    "inflation",
    "trade",
    "unemployment",
    "urban",
    "va_score",
    "cc_score",
    "ge_score"
]

# --------------------------------------------------
# 3. Function to aggregate one country's imputed files
# --------------------------------------------------

def aggregate_country_imputations(country_name, folder_path):
    csv_files = sorted(folder_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found for {country_name} in {folder_path}")

    all_dfs = []

    for i, file in enumerate(csv_files, start=1):
        df = pd.read_csv(file)

        required_cols = ["year"] + variables

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{file} is missing required column: {col}")

        temp = df[["year"] + variables].copy()
        temp["imputation"] = i
        temp["country"] = country_name

        all_dfs.append(temp)

    stacked = pd.concat(all_dfs, ignore_index=True)

    aggregated = (
        stacked
        .groupby(["country", "year"], as_index=False)[variables]
        .mean()
    )

    return aggregated

# --------------------------------------------------
# 4. Aggregate all countries into one pooled dataset
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
# 5. Create correlation matrix
# --------------------------------------------------

corr_matrix = pooled_aggregated[variables].corr()

print(corr_matrix)

# --------------------------------------------------
# 6. Plot correlation heatmap
# --------------------------------------------------

plt.figure(figsize=(12, 10))

heatmap = plt.imshow(
    corr_matrix,
    aspect="auto",
    vmin=-1,
    vmax=1
)

# Color bar
cbar = plt.colorbar(heatmap)
cbar.set_label("Correlation Coefficient", rotation=270, labelpad=20)

# Axis labels
plt.xticks(
    ticks=np.arange(len(variables)),
    labels=variables,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(variables)),
    labels=variables
)

plt.title("Correlation Heatmap of Key Variables")
plt.tight_layout()

plt.savefig("correlation_heatmap_pooled_dataset.png", dpi=300, bbox_inches="tight")
plt.show()