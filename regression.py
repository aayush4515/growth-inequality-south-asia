# code to run regression on imputed dataset
from constants import economies
import statsmodels.formula.api as sm
import pandas as pd
from pathlib import Path

impute_cols = [
    "year",
    "education",
    "fdi",
    "gdp_growth",
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

# function to created an OLS model for one country
def createModelForOneCountry(df):
    # formula to pass into the model
    formula = 'palma_ratio~gdp_growth+education+fdi+log_gdp_pc+industry+inflation+trade+unemployment+urban+va_score+cc_score+ge_score'

    # regression for individual country
    model = sm.ols(formula=formula, data=df).fit()

    return model

# run for all available datasets for the country
def runRegression(economy):
    # store the result of all the datasets in a results array
    results = []

    # path to the data directory
    data_directory_str = f"IMPUTED_DATA/{economy}"
    data_dir = Path(data_directory_str)

    # iterate over all the csvs in that directory
    for file in sorted(data_dir.glob("*.csv")):
        # read the csv
        df = pd.read_csv(file)

        # train the model on that csv
        countryModel = createModelForOneCountry(df)

        # append the model to the results array
        results.append(countryModel)

    return results

# now run regression on all the economies and print summaries
for economy in economies:
    resultsPerCountry = runRegression(economy)

    # print summary of each country
    for result_num, result in enumerate(resultsPerCountry, start=1):
        print(f"\nResults {result_num} for {economy}:")
        print(result.summary())


