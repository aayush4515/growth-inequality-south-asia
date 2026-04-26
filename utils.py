from constants import indicators, countries
import pandas as pd
import wbgapi as wb

# function to extract data for a particular country
def extractCountryData(country):
    df = wb.data.DataFrame(
        indicators,
        economy=country,
        time=range(1995, 2026),
        labels=True
    ).reset_index()

    return df

# function to save the dataframe to an excel and csv file, country too if provided
def saveToCSVandExcel(df, country=None):
    df.to_csv(f"RAW_DATA_CSV/{country}_raw_data.csv", index=False)
    df.to_excel(f"RAW_DATA_XLSX/{country}_raw_data.xlsx", index=False)

# function to convert data into panel format
def convertToPanelFormat(df, country):
    df_long = df.melt(
        id_vars=["series", "Series"],
        var_name="year",
        value_name="value"
    )

    df_long["year"] = df_long["year"].str.replace("YR", "").astype(int)

    df_final = df_long.pivot_table(
        index=["year"],
        columns="series",
        values="value"
    ).reset_index()

    df_final.insert(0, "country", country)

    return df_final