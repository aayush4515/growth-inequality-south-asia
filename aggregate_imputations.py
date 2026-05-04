# aggregate the imputations across different country into a single panel data
# for example: aggregate database 1 of all five countries into a single country
# create five all-countries databases in total

"""
Aggregate Country-Level Imputed Datasets into Full Panel Datasets

This script combines the country-wise imputed datasets into full all-country
panel datasets.

Before running this script, multiple imputation was performed separately for
each country/economy. For example, each country has several imputed datasets
saved in its own folder:

    IMPUTED_DATA/BGD/BGD_imputed_1.csv
    IMPUTED_DATA/BGD/BGD_imputed_2.csv
    ...
    IMPUTED_DATA/IND/IND_imputed_1.csv
    IMPUTED_DATA/IND/IND_imputed_2.csv
    ...

The purpose of this script is not to average the imputed datasets. Instead, it
combines datasets by matching imputation number across countries. For example:

    BGD_imputed_1
    IND_imputed_1
    NPL_imputed_1
    PAK_imputed_1
    LKA_imputed_1

are stacked together into one complete panel dataset:

    All_Countries_imputed_1.csv

The same process is repeated for imputation 2, imputation 3, and so on. If each
country has 5 imputed datasets, this script creates 5 full all-country panel
datasets in total.

The main function, `aggregateImps(economies)`, works in three steps:

1. It loops through each economy's folder inside `IMPUTED_DATA/`.
2. It loads that economy's imputed CSV files into a list and stores the list in
   a dictionary where the key is the economy name.
3. It uses `zip(*all_agg_datasets.values())` to group together imputed datasets
   with the same index across all countries, then concatenates them into full
   panel datasets.

The final output files are saved in:

    IMPUTED_DATA/All_Countries/

These aggregated panel datasets can later be used for regression analysis. The
same regression model should be run separately on each all-country imputed
dataset, and the results should be pooled using Rubin's Rules.
"""

from constants import countries, economies
from pathlib import Path
import pandas as pd
import os

def aggregateImps(economies):
    # create lists of datasets for each economy that would store all imputed datasets for that economy

    # stores all aggregated datasets
    all_agg_datasets = dict()
    for economy in economies:
        economy_dir = Path(f"IMPUTED_DATA/{economy}")
        # get the number of datasets in that economy's directory
        num_files = sum(1 for item in economy_dir.iterdir() if item.is_file())

        # aggregate all the files into a single dataset for that economy
        economy_dataset = []
        for i in range(num_files):
            df = pd.read_csv(f"IMPUTED_DATA/{economy}/{economy}_imputed_{i+1}.csv")
            economy_dataset.append(df)

        # append that particular economy's dataset to the global dataset all_agg_datasets
        all_agg_datasets[economy] = economy_dataset


    # after all the datasets' list are aggregated into a single global dataset
    # aggregate per economy per index

    combined_datasets = []

    for dfs in zip(*all_agg_datasets.values()):
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_datasets.append(combined_df)

    # now save all aggregated datasets in corresponding CSVs
    os.makedirs("IMPUTED_DATA/All_Countries", exist_ok=True)
    for dataset_num, dataset_df in enumerate(combined_datasets, start=1):
        dataset_df.to_csv(f"IMPUTED_DATA/All_Countries/All_Countries_imputed_{dataset_num}.csv", index=False)

# run the aggregate function
aggregateImps(economies)