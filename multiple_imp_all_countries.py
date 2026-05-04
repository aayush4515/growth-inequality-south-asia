# aggregate the raw data of all countries into a single dataframe and perform MICE

# FIXME: INCOMPLETE, NEEDS MODIFICATION

from constants import countries, economies, impute_cols
import pandas as pd
from statsmodels.imputation.mice import MICEData

# function to perform multiple imputation on all countries
def runMultipeImputationAllCountries(numDatasets, numImputations):
    # load the aggregated dataset
    df_original = pd.read_csv(f"CSV_DATA/RAW_DATA/All_Countries_Raw.csv")

    # list of imputed datasets
    imputed_datasets = []

    for i in range(numDatasets):
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


