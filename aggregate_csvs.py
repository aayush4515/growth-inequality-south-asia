# aggregate all the raw .csvs into a single csv

from constants import economies, countries
import pandas as pd

# stores all the country data into a list
dfs = []
for economy in economies:
    df = pd.read_csv(f"CSV_DATA/RAW_DATA/{economy}_raw_data.csv")

    dfs.append(df)

# combine all the dataframe into one
df_aggregated = pd.concat(dfs, ignore_index=True)

# save it to as CSV
df_aggregated.to_csv(f"CSV_DATA/RAW_DATA/All_Countries_Raw.csv", index=False)