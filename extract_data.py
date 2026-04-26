import wbgapi as wb
import pandas as pd
from constants import indicators, countries
from utils import extractCountryData, saveToCSVandExcel, convertToPanelFormat

# extract data for each of the five countires individualy
for country in countries:
    df = extractCountryData(country)

    # convert the data into panel format
    df_panel = convertToPanelFormat(df, country)

    # save data
    saveToCSVandExcel(df_panel, country)