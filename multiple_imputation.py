# run multiple imputation country-wise and create 5 datasets per country

from constants import countries, economies, impute_cols
import pandas as pd
from statsmodels.imputation.mice import MICEData
import os
import time

# function to run multiple imputation per country/economy
def runMultipleImputation(economy, numDatasets, numImputations):
    # load the raw dataset for the country
    df_original = pd.read_csv(f"CSV_DATA/RAW_DATA/{economy}_raw_data.csv")

    # run 'numImputation' imputation cycles to create 'numDatasets' datasets
    imputed_datasets = []

    for i in range(numDatasets):    # create 'numDatasets' imputed datasets
        # working dataframe
        df_working = df_original.copy()

        # create a MICE dataset using impute_cols
        imp_data = MICEData(df_working[impute_cols].copy())

        # run 'numImputations' imputation cycles
        for j in range(numImputations):
            imp_data.update_all()

        # replace the original numeric columns with updated values
        df_working[impute_cols] = imp_data.data

        # append the imputed dataset to imputed_datasets
        imputed_datasets.append(df_working.copy())

    # return the imputed datasets for that economy
    return imputed_datasets

# create a directory to hold the imputed data
os.makedirs("IMPUTED_DATA", exist_ok=True)

start = time.time()

# run multiple imputation per economy and save them to the folder IMPUTED_DATA
for economy in economies:
    # datasets per economy, 5 datasets with 20 imputation cycles each
    datasets = runMultipleImputation(economy, 5, 20)

    # save each dataset for each country
    for dataset_num, data in enumerate(datasets, start=1):
        data.to_csv(f"IMPUTED_DATA/{economy}/{economy}_imputed_{dataset_num}.csv", index=False)

end = time.time()

print(f"Execution time: {end - start:.2f} seconds")