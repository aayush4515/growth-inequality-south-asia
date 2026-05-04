# code to run regression on imputed dataset
from constants import economies
import statsmodels.formula.api as sm
import pandas as pd
from pathlib import Path

# function to created an OLS model for one country
def createModel(df):
    # formula to pass into the model
    formula = 'palma_ratio~gdp_growth+education+fdi+log_gdp_pc+industry+inflation+trade+unemployment+urban+va_score+cc_score+ge_score+C(country)+C(year)'

    # regression for individual country
    model = sm.ols(formula=formula, data=df).fit()

    return model

# run for all countries for each dataset
def runRegression():
    # store the result of all the datasets in a results array
    results = []

    # path to the data directory
    data_directory_str = f"IMPUTED_DATA/All_Countries"
    data_dir = Path(data_directory_str)

    # iterate over all the csvs in that directory
    for file in sorted(data_dir.glob("*.csv")):
        # read the csv
        df = pd.read_csv(file)

        # train the model on that csv
        model = createModel(df)

        # append the model to the results array
        results.append(model)

    return results

# run the pooled regression
results = runRegression()

# print summary of each country
for result_num, result in enumerate(results, start=1):
    print(f"Pooled result of dataset {result_num} for all countries:")
    print(result.summary())


