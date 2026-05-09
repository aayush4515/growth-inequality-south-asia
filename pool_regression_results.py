import os
import numpy as np
from scipy import stats
from pooled_regression import runRegression

# run the regression and save it in results
results = runRegression()

def pool_variable(results, variable_name):
    """
    Pool one regression coefficient across multiple imputed datasets
    using Rubin's Rules.
    """

    m = len(results)

    # Get coefficient for this variable from each model
    coefficients = np.array([
        model.params[variable_name] for model in results
    ])

    # Get standard error for this variable from each model
    standard_errors = np.array([
        model.bse[variable_name] for model in results
    ])

    # Convert standard errors to variances
    variances = standard_errors ** 2

    # Rubin's Rules

    # 1. Pooled coefficient
    pooled_coefficient = coefficients.mean()

    # 2. Average within-imputation variance
    within_variance = variances.mean()

    # 3. Between-imputation variance
    between_variance = coefficients.var(ddof=1)

    # 4. Total variance
    total_variance = within_variance + (1 + 1 / m) * between_variance

    # 5. Pooled standard error
    pooled_standard_error = np.sqrt(total_variance)

    # 6. z-score
    z_score = pooled_coefficient / pooled_standard_error

    # 7. final two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # 8. 95% confidence interval
    ci_lower = pooled_coefficient - 1.96 * pooled_standard_error
    ci_upper = pooled_coefficient + 1.96 * pooled_standard_error

    return {
        "variable": variable_name,
        "pooled_coefficient": pooled_coefficient,
        "pooled_standard_error": pooled_standard_error,
        "z_score": z_score,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "individual_coefficients": coefficients,
        "individual_standard_errors": standard_errors
    }


gdp_growth_result = pool_variable(results, "gdp_growth")


print("\nFinal pooled result for GDP Growth")
print("=" * 50)
print(f"Coefficient: {gdp_growth_result['pooled_coefficient']:.6f}")
print(f"Standard Error: {gdp_growth_result['pooled_standard_error']:.6f}")
print(f"z-score: {gdp_growth_result['z_score']:.6f}")
print(f"p-value: {gdp_growth_result['p_value']:.6f}")
print(
    f"95% CI: "
    f"[{gdp_growth_result['ci_lower']:.6f}, "
    f"{gdp_growth_result['ci_upper']:.6f}]"
)


# Save final pooled result to file
os.makedirs("REGRESSION_SUMMARY/POOLED", exist_ok=True)

with open("REGRESSION_SUMMARY/POOLED/final_pooled_gdp_growth_result.txt", "w") as file:
    file.write("FINAL POOLED RESULT FOR GDP GROWTH\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Coefficient: {gdp_growth_result['pooled_coefficient']:.6f}\n")
    file.write(f"Standard Error: {gdp_growth_result['pooled_standard_error']:.6f}\n")
    file.write(f"z-score: {gdp_growth_result['z_score']:.6f}\n")
    file.write(f"p-value: {gdp_growth_result['p_value']:.6f}\n")
    file.write(
        f"95% CI: "
        f"[{gdp_growth_result['ci_lower']:.6f}, "
        f"{gdp_growth_result['ci_upper']:.6f}]\n"
    )

    file.write("\nIndividual results from each imputed dataset:\n")
    file.write("-" * 60 + "\n")

    for i, coef in enumerate(gdp_growth_result["individual_coefficients"], start=1):
        se = gdp_growth_result["individual_standard_errors"][i - 1]
        file.write(f"Dataset {i}: coefficient = {coef:.6f}, SE = {se:.6f}\n")