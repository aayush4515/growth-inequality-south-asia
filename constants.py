# list of indicators
indicators = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NY.GDP.PCAP.KD": "gdp_pc",
    "SP.URB.TOTL.IN.ZS": "urban",
    "NV.IND.TOTL.ZS": "industry",
    "SE.SEC.ENRR": "education",
    "BX.KLT.DINV.WD.GD.ZS": "fdi",
    "NE.TRD.GNFS.ZS": "trade",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment"
}

# economies to fetch data from
economies = ["BGD", "IND", "NPL", "PAK", "LKA"]

# countries
countries = ["Bangladesh", "India", "Nepal", "Pakistan", "Srilanka"]

# columns to use in the imputation model
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
