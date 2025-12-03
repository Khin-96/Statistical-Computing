import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- 1. Data Loading/Preparation (Simulated birthwt data) ---
# The original 'birthwt' data has 189 rows. We create a simulated, 
# but structurally similar, dataset based on the R summary provided.

# Data structure based on summary(birthwt) and help(birthwt) 
data = {
    'low': np.random.randint(0, 2, 189), # low birth weight (0/1)
    'age': np.random.randint(14, 46, 189), # mother's age
    'lwt': np.random.randint(80, 251, 189), # mother's weight (lbs)
    'race': np.random.choice([1, 2, 3], 189, p=[0.5, 0.2, 0.3]), # race (1, 2, 3)
    'smoke': np.random.choice([0, 1], 189, p=[0.6, 0.4]), # smoke during pregnancy (0/1)
    'bwt': np.random.randint(709, 4991, 189), # birth weight (grams)
}
# We manually inject the outlier for replication purposes:
data['age'][0] = 45 # Oldest mother
data['bwt'][0] = 4990 # Heaviest child

df = pd.DataFrame(data)

# Replicate the R-style data prep: ensuring 'smoke' is categorical for modeling
df['mother_smokes'] = df['smoke'].astype('category').cat.rename_categories({0: 'No', 1: 'Yes'})
df['birthwt_grams'] = df['bwt']

print("--- DataFrame Summary (Python Equivalent) ---")
print(df[['birthwt_grams', 'age', 'mother_smokes']].head())
print("\n" + "-"*50 + "\n")

# --- 2. Simple Linear Model: bwt ~ age ---
# This uses the R-style formula interface (ols)
linear_model_2 = ols('birthwt_grams ~ age', data=df).fit()

print("--- Simple Linear Model: bwt ~ age (All Data) ---")
print(linear_model_2.summary())
print("\n" + "-"*50 + "\n")

# --- 3. Outlier Removal and Simple Linear Model ---
# Remove the outlier row where age > 40
df_noout = df[df['age'] <= 40].copy()

linear_model_3 = ols('birthwt_grams ~ age', data=df_noout).fit()

print("--- Simple Linear Model: bwt ~ age (Outlier Removed) ---")
print(linear_model_3.summary())
print("\n" + "-"*50 + "\n")


# --- 4. Multiple Linear Model: bwt ~ smoke + age ---
# The formula will automatically handle the categorical variable 'mother_smokes'
linear_model_3a = ols('birthwt_grams ~ mother_smokes + age', data=df_noout).fit()

print("--- Multiple Linear Model: bwt ~ smoke + age (Outlier Removed) ---")
print(linear_model_3a.summary())