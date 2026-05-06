# this code gives summary statistics, correlation matrix, and VIF table
# for the pooled all-country imputed datasets

from pathlib import Path
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
# do NOT include the dependent variable palma_ratio
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

# path to pooled all-country imputed datasets
data_dir_pooled = Path("IMPUTED_DATA/All_Countries")

# lists to store tables from each imputed dataset
summary_tables = []
correlation_tables = []
vif_tables = []


# function to create descriptive statistics table
def create_summary_table(df):
    summary = pd.DataFrame(index=main_variables)

    summary["mean"] = df[main_variables].mean()
    summary["median"] = df[main_variables].median()
    summary["std_dev"] = df[main_variables].std()
    summary["std_error"] = df[main_variables].std() / np.sqrt(df[main_variables].count())
    summary["min"] = df[main_variables].min()
    summary["max"] = df[main_variables].max()

    return summary


# function to create correlation matrix
def create_correlation_matrix(df):
    correlation_matrix = df[main_variables].corr()
    return correlation_matrix


# function to create VIF table
def create_vif_table(df):
    # only use independent/control variables
    X = df[x_variables].copy()

    # remove any rows with missing values just in case
    X = X.dropna()

    # add constant/intercept for VIF calculation
    X = sm.add_constant(X)

    vif_table = pd.DataFrame()
    vif_table["variable"] = X.columns
    vif_table["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_table.set_index("variable")


# loop through all imputed all-country datasets
for file in sorted(data_dir_pooled.glob("*.csv")):
    df = pd.read_csv(file)

    # Table 1: descriptive statistics
    summary_tables.append(create_summary_table(df))

    # Table 2: correlation matrix
    correlation_tables.append(create_correlation_matrix(df))

    # Table 3: VIF table
    vif_tables.append(create_vif_table(df))


# average tables across the 5 imputed datasets
pooled_summary = sum(summary_tables) / len(summary_tables)
pooled_correlation = sum(correlation_tables) / len(correlation_tables)
pooled_vif = sum(vif_tables) / len(vif_tables)


# print results
print("\nTable 1: Descriptive Statistics")
print(pooled_summary)

print("\nTable 2: Correlation Matrix")
print(pooled_correlation)

print("\nTable 3: VIF Table")
print(pooled_vif)


# save outputs
output_dir = Path("SUMMARY_TABLES")
output_dir.mkdir(exist_ok=True)

pooled_summary.to_csv(output_dir / "all_countries_table_1_descriptive_statistics.csv")
pooled_correlation.to_csv(output_dir / "all_countries_table_2_correlation_matrix.csv")
pooled_vif.to_csv(output_dir / "all_countries_table_3_vif_table.csv")