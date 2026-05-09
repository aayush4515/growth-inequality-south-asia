# This file contains the functions calls for the entire data extraction and analysis workflow

# 1. Clean the WID data, use clean.py [NOTE: need to manually extract this data]

# 2. Clean the WGI data, use clean_wgi_data.py [NOTE: need to manually extract this data]

# 3. Compute Palma Ratios for all the economies, use compute_palma.py

# 4. Extract all the required data and save to CSV_DATA/RAW_DATA, use extract_data.py

# 5. Aggregate all per-country CSVs to one single CSV, use aggregate_csvs.py

# 6. Run multiple imputations on all country's raw data, save to IMPUTED_DATA, use multiple_imputation.py

# 7. Aggregate the per-country imputations together, use aggregate_imputations.py

# 8. Run regression on the all-countries imputaed datasets and save the results to REGRESSION_SUMMARY/POOLED, use pool_regression_results.py
#[NOTE: Can use the code inside the main scope of pooled_regression.py to save the per-dataset summary]



