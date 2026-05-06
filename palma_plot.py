import pandas as pd
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
# 2. Function to aggregate one country's imputed files
# --------------------------------------------------

def aggregate_country_imputations(country_name, folder_path):
    csv_files = sorted(folder_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found for {country_name} in {folder_path}")

    all_dfs = []

    for i, file in enumerate(csv_files, start=1):
        df = pd.read_csv(file)

        required_cols = ["year", "palma_ratio"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{file} is missing required column: {col}")

        temp = df[["year", "palma_ratio"]].copy()
        temp["imputation"] = i
        temp["country"] = country_name

        all_dfs.append(temp)

    stacked = pd.concat(all_dfs, ignore_index=True)

    aggregated = (
        stacked
        .groupby(["country", "year"], as_index=False)
        .agg(avg_palma_ratio=("palma_ratio", "mean"))
    )

    return aggregated

# --------------------------------------------------
# 3. Aggregate all countries
# --------------------------------------------------

country_results = []

for country_name, folder_path in countries.items():
    country_avg = aggregate_country_imputations(country_name, folder_path)
    country_results.append(country_avg)

palma_ratio_aggregated = pd.concat(country_results, ignore_index=True)
palma_ratio_aggregated = palma_ratio_aggregated.sort_values(["country", "year"])

# --------------------------------------------------
# 4. Find top 2 countries by average GDP growth
# --------------------------------------------------

top_2_countries = (
    palma_ratio_aggregated
    .groupby("country")["avg_palma_ratio"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

print("Top 2 countries by average GDP growth:")
print(top_2_countries.tolist())

top_2_df = palma_ratio_aggregated[
    palma_ratio_aggregated["country"].isin(top_2_countries)
]

# --------------------------------------------------
# 5. Plot only the top 2 countries
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for country in top_2_df["country"].unique():
    country_data = top_2_df[top_2_df["country"] == country]

    plt.plot(
        country_data["year"],
        country_data["avg_palma_ratio"],
        marker="o",
        linewidth=2,
        label=country
    )

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Year")
plt.ylabel("Average Palma Ratio (%)")
plt.title("Palma Ratio Trends for Top 3 Countries\nAveraged Across 20 Imputed Datasets")
plt.legend(title="Country")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("top_2_palma_ratio_countries.png", dpi=300, bbox_inches="tight")
plt.show()