# code to run regression on imputed dataset
from constants import economies
import statsmodels.formula.api as sm
import pandas as pd
import os
from pathlib import Path

# function to created an OLS model for one country
def createModel(df):
    # formula to pass into the model
    formula = 'palma_ratio~gdp_growth+education+fdi+log_gdp_pc+industry+inflation+trade+unemployment+urban+va_score+cc_score+ge_score+C(country)+C(year)'

    # regression for individual country
    model = sm.ols(formula=formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={"groups": df["country"]}
    )

    # model = sm.ols(formula=formula, data=df).fit()

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

if __name__ == '__main__':

    # print summary of each pooled dataset

    # create folder once
    os.makedirs("REGRESSION_SUMMARY/POOLED", exist_ok=True)
    saving_dir = "REGRESSION_SUMMARY/POOLED"

    # one combined file
    file_path = f"{saving_dir}/pooled_regression_summaries.txt"

    # clear the file first so old results do not stay there
    with open(file_path, "w") as file:
        file.write("POOLED REGRESSION SUMMARIES FOR ALL IMPUTED DATASETS\n")
        file.write("=" * 80 + "\n\n")

    # append each summary to the same file
    for result_num, result in enumerate(results, start=1):
        print(f"Pooled result of dataset {result_num} for all countries:")

        with open(file_path, "a") as file:
            file.write(f"\n\nPooled result of dataset {result_num} for all countries:\n")
            file.write("=" * 80 + "\n")
            file.write(result.summary().as_text())
            file.write("\n" + "=" * 80 + "\n")

        print(result.summary())


