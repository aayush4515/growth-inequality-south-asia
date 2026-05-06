# this code gives summary statistics, correlation matrix, and VIF table
# for each country's imputed datasets

from pathlib import Path
from constants import economies
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# variables to use in descriptive statistics and correlation matrix
main_variables = [
    "palma_ratio",
    "gdp_growth",
    "log_gdp_pc",
    "education",
    "fdi",
    "industry",
    "inflation",
    "trade",
    "unemployment",
    "urban",
    "va_score",
    "cc_score",
    "ge_score"
]

# independent/control variables for VIF
# do NOT include dependent variable palma_ratio
x_variables = [
    "gdp_growth",
    "log_gdp_pc",
    "education",
    "fdi",
    "industry",
    "inflation",
    "trade",
    "unemployment",
    "urban",
    "va_score",
    "cc_score",
    "ge_score"
]

def create_summary_table(df):
    summary = pd.DataFrame(index=main_variables)

    summary["mean"] = df[main_variables].mean()
    summary["median"] = df[main_variables].median()
    summary["std_dev"] = df[main_variables].std()
    summary["std_error"] = df[main_variables].std() / np.sqrt(df[main_variables].count())
    summary["min"] = df[main_variables].min()
    summary["max"] = df[main_variables].max()

    return summary


def create_correlation_matrix(df):
    return df[main_variables].corr()


def create_vif_table(df):
    X = df[x_variables].copy()

    # remove any missing rows just in case
    X = X.dropna()

    # add intercept
    X = sm.add_constant(X)

    vif_table = pd.DataFrame()
    vif_table["variable"] = X.columns
    vif_table["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    # remove const because we usually do not interpret it
    vif_table = vif_table[vif_table["variable"] != "const"]

    return vif_table.set_index("variable")


# output folder
output_dir = Path("SUMMARY_TABLES/Per_Country")
output_dir.mkdir(parents=True, exist_ok=True)


# loop through each country/economy
for economy in economies:
    print(f"\n==============================")
    print(f"Summary tables for {economy}")
    print(f"==============================")

    data_dir = Path(f"IMPUTED_DATA/{economy}")

    summary_tables = []
    correlation_tables = []
    vif_tables = []

    # loop through that country's imputed datasets
    for file in sorted(data_dir.glob(f"{economy}_imputed_*.csv")):
        df = pd.read_csv(file)

        summary_tables.append(create_summary_table(df))
        correlation_tables.append(create_correlation_matrix(df))
        vif_tables.append(create_vif_table(df))

    # average across that country's 5 imputed datasets
    country_summary = sum(summary_tables) / len(summary_tables)
    country_correlation = sum(correlation_tables) / len(correlation_tables)
    country_vif = sum(vif_tables) / len(vif_tables)

    # print results
    print("\nTable 1: Descriptive Statistics")
    print(country_summary)

    print("\nTable 2: Correlation Matrix")
    print(country_correlation)

    print("\nTable 3: VIF Table")
    print(country_vif)

    # create country-specific output folder
    country_output_dir = output_dir / economy
    country_output_dir.mkdir(parents=True, exist_ok=True)

    # save outputs
    country_summary.to_csv(country_output_dir / f"{economy}_table_1_descriptive_statistics.csv")
    country_correlation.to_csv(country_output_dir / f"{economy}_table_2_correlation_matrix.csv")
    country_vif.to_csv(country_output_dir / f"{economy}_table_3_vif_table.csv")