# Economic Growth and Income Inequality in South Asia

This project studies the relationship between economic growth and income inequality in selected South Asian countries. The main research question is whether economic growth eventually reduces income inequality in South Asia, or whether inequality continues to persist because of structural and institutional factors.

The project builds a panel dataset for five countries:

- Bangladesh
- India
- Nepal
- Pakistan
- Sri Lanka

The dataset combines macroeconomic indicators, governance indicators, and inequality measures. The final goal is to prepare regression-ready panel datasets that can be used to test the relationship between growth, development, governance, and inequality.

---

## Research Focus

The project focuses on the following broad question:

> Does economic growth reduce income inequality in South Asian countries over time, or does inequality persist despite growth?

The dependent variable used for inequality analysis is the **Palma ratio**, which compares the income share of the top 10% to the income share of the bottom 40%.

The explanatory and control variables include indicators such as:

- GDP growth
- GDP per capita
- Log GDP per capita
- Education enrollment
- Foreign direct investment
- Industry share
- Inflation
- Trade openness
- Unemployment
- Urbanization
- Voice and accountability
- Control of corruption
- Government effectiveness

---

## Project Structure

```text
econ-research/
│
├── CSV_DATA/
│   └── RAW_DATA/
│       └── Raw country-level CSV datasets
│
├── XLSX_DATA/
│   └── RAW_DATA/
│       └── Raw Excel files
│
├── WID_DATA_10%/
│   └── World Inequality Database top 10% income share files
│
├── WID_DATA_40%/
│   └── World Inequality Database bottom 40% income share files
│
├── PALMA_RATIOS/
│   └── Computed Palma ratio files
│
├── WGI_DATA/
│   └── Raw Worldwide Governance Indicators data
│
├── WGI_DATA_CLEANED/
│   └── Cleaned governance indicator files
│
├── IMPUTED_DATA/
│   ├── BGD/
│   ├── IND/
│   ├── NPL/
│   ├── PAK/
│   ├── LKA/
│   └── All_Countries/
│       └── Aggregated all-country imputed panel datasets
│
├── aggregate_csvs.py
├── aggregate_imputations.py
├── clean.py
├── clean_wgi_data.py
├── compute_palma.py
├── constants.py
├── extract_data.py
├── inetrpolate.py
├── multiple_imp_all_countries.py
├── multiple_imputation.py
├── requirements.txt
└── utils.py
```

---

## Data Sources

This project uses data from several public economic and governance data sources.

### World Bank Indicators

The project uses `wbgapi` to fetch country-level macroeconomic indicators from the World Bank. The indicators include:

| Indicator | Variable Name |
|---|---|
| GDP growth | `gdp_growth` |
| GDP per capita | `gdp_pc` |
| Urban population share | `urban` |
| Industry share | `industry` |
| Secondary school enrollment | `education` |
| Foreign direct investment | `fdi` |
| Trade openness | `trade` |
| Inflation | `inflation` |
| Unemployment | `unemployment` |

### World Inequality Database

The World Inequality Database files are used to calculate the Palma ratio:

```text
Palma Ratio = Top 10% income share / Bottom 40% income share
```

The project stores top 10% and bottom 40% data separately, then computes the Palma ratio using `compute_palma.py`.

### Worldwide Governance Indicators

The project also includes cleaned governance indicators such as:

- Voice and accountability: `va_score`
- Control of corruption: `cc_score`
- Government effectiveness: `ge_score`

---

## Methodology

### 1. Data Extraction

The script `extract_data.py` is used to collect macroeconomic data for the selected South Asian countries. The country codes used in this project are:

```python
economies = ["BGD", "IND", "NPL", "PAK", "LKA"]
```

These correspond to Bangladesh, India, Nepal, Pakistan, and Sri Lanka.

---

### 2. Data Cleaning

The project includes several cleaning scripts:

- `clean.py` cleans and prepares country-level data.
- `clean_wgi_data.py` cleans governance indicator data.
- `compute_palma.py` calculates Palma ratios from top 10% and bottom 40% income share files.
- `aggregate_csvs.py` combines raw country-level CSV files into a larger all-country dataset.

---

### 3. Missing Data Strategy

Because the dataset contains missing values across several indicators and years, this project uses **Multiple Imputation by Chained Equations (MICE)**.

The main country-wise imputation script is:

```text
multiple_imputation.py
```

The imputation model uses the following variables:

```python
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
```

The `year` variable is included so that the imputation model can account for time trends within each country.

For each country, the script creates multiple imputed datasets. For example:

```text
IMPUTED_DATA/BGD/BGD_imputed_1.csv
IMPUTED_DATA/BGD/BGD_imputed_2.csv
IMPUTED_DATA/BGD/BGD_imputed_3.csv
IMPUTED_DATA/BGD/BGD_imputed_4.csv
IMPUTED_DATA/BGD/BGD_imputed_5.csv
```

This is repeated for all five countries.

---

### 4. Aggregating Imputed Datasets into Panel Data

After country-wise MICE is completed, `aggregate_imputations.py` combines the country-level imputed datasets into full all-country panel datasets.

The script does **not** average the imputed datasets. Instead, it combines matching imputation numbers across countries.

For example:

```text
BGD_imputed_1
IND_imputed_1
NPL_imputed_1
PAK_imputed_1
LKA_imputed_1
```

are stacked together into:

```text
All_Countries_imputed_1.csv
```

The same process is repeated for the second, third, fourth, and fifth imputed datasets.

Final outputs are saved in:

```text
IMPUTED_DATA/All_Countries/
```

This creates multiple complete panel datasets that can later be used for regression analysis.

---

## Why Multiple Imputation?

The project uses multiple imputation instead of simply dropping missing values or filling missing values with averages.

This is important because:

1. Dropping rows would reduce the sample size.
2. Mean imputation would ignore country-specific and time-based patterns.
3. Multiple imputation better preserves uncertainty around missing values.
4. Running regressions across multiple imputed datasets allows final results to be pooled using Rubin's Rules.

The correct workflow is:

```text
Country-wise raw data
        ↓
Country-wise MICE
        ↓
Multiple imputed datasets per country
        ↓
Aggregate matching imputation numbers across countries
        ↓
Multiple complete all-country panel datasets
        ↓
Regression on each completed dataset
        ↓
Pool regression results using Rubin's Rules
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aayush4515/econ-research.git
cd econ-research
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

The project uses the following main Python libraries:

```text
wbgapi
pandas
openpyxl
numpy
statsmodels
```

---

## How to Run the Project

A typical workflow is:

### 1. Extract World Bank data

```bash
python extract_data.py
```

### 2. Clean the data

```bash
python clean.py
```

### 3. Clean governance data

```bash
python clean_wgi_data.py
```

### 4. Compute Palma ratios

```bash
python compute_palma.py
```

### 5. Run country-wise multiple imputation

```bash
python multiple_imputation.py
```

### 6. Aggregate imputed datasets into all-country panel datasets

```bash
python aggregate_imputations.py
```

After this step, the regression-ready panel datasets will be available in:

```text
IMPUTED_DATA/All_Countries/
```

---

## Current Status

The project currently includes:

- Data extraction scripts
- Data cleaning scripts
- Palma ratio calculation
- Governance data cleaning
- Country-wise MICE imputation
- Aggregation of imputed country-level datasets into all-country panel datasets
- Regression-ready imputed panel outputs

Future work may include:

- Running panel regression models
- Adding two-way fixed effects models
- Pooling regression outputs using Rubin's Rules
- Creating final tables and visualizations for the research paper

---

## Notes

This project is part of an economics research workflow focused on growth and inequality in South Asia. The code is written in Python and organized around reproducible data preparation, missing data treatment, and panel dataset construction.

---

## Contributors

[Aayush Acharya](https://github.com/aayush4515) <br />
[Inaya Zoraiz](https://github.com/izora0483)