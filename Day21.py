import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm

np.random.seed(42)

# ======================================================
# 1️⃣ CREATE SYNTHETIC BUSINESS DATASET
# ======================================================

data = {
    "Age": np.random.randint(20, 60, 200),
    "Income": np.random.randint(30000, 120000, 200),
    "Sales": np.random.randint(5000, 50000, 200),
    "Gender": np.random.choice(["Male", "Female"], 200),
    "Department": np.random.choice(["IT", "HR", "Sales"], 200)
}

df = pd.DataFrame(data)

# Introduce some missing values artificially
df.loc[np.random.choice(df.index, 10), "Income"] = np.nan

# ======================================================
# 2️⃣ DATASET OVERVIEW
# ======================================================

print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())

# ======================================================
# 3️⃣ MISSING VALUE ANALYSIS
# ======================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Percentage:")
print((df.isnull().sum() / len(df)) * 100)

# Fill missing Income with median
df["Income"].fillna(df["Income"].median(), inplace=True)

# ======================================================
# 4️⃣ DUPLICATE CHECK
# ======================================================

duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# ======================================================
# 5️⃣ DISTRIBUTION OF NUMERICAL FEATURES
# ======================================================

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ======================================================
# 6️⃣ CATEGORICAL ANALYSIS
# ======================================================

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts())

    plt.figure(figsize=(5,3))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot - {col}")
    plt.show()

# ======================================================
# 7️⃣ CORRELATION ANALYSIS
# ======================================================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ======================================================
# 8️⃣ OUTLIER DETECTION (IQR METHOD)
# ======================================================

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\nOutliers in {col}: {len(outliers)}")

    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Detection - {col}")
    plt.show()

# ======================================================
# 9️⃣ CENTRAL TENDENCY
# ======================================================

for col in numerical_cols:
    print(f"\nColumn: {col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:", df[col].mode()[0])

# ======================================================
# 🔟 VARIANCE & STANDARD DEVIATION
# ======================================================

for col in numerical_cols:
    print(f"\nColumn: {col}")
    print("Variance:", df[col].var())
    print("Std Dev:", df[col].std())

# ======================================================
# 1️⃣1️⃣ PROBABILITY ANALYSIS
# ======================================================

threshold = df["Sales"].mean()
prob = (df["Sales"] > threshold).mean()

print("\nProbability Sales > Mean:", prob)

# ======================================================
# 1️⃣2️⃣ HYPOTHESIS TESTING (Income by Gender)
# ======================================================

group1 = df[df["Gender"] == "Male"]["Income"]
group2 = df[df["Gender"] == "Female"]["Income"]

t_stat, p_value = ttest_ind(group1, group2)

print("\nT-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Statistically significant difference between male and female income.")
else:
    print("No statistically significant difference between male and female income.")


mean = df["Income"].mean()
std = df["Income"].std()
n = len(df)

z = norm.ppf(0.975)
margin = z * (std / np.sqrt(n))

print("\n95% Confidence Interval for Income:")
print(mean - margin, "to", mean + margin)

# ======================================================
# 1️⃣4️⃣ FINAL SUMMARY INSIGHTS
# ======================================================

print("\n----- KEY INSIGHTS -----")
print("1. Dataset contains", df.shape[0], "rows after cleaning.")
print("2. Income distribution shows moderate variance.")
print("3. Sales probability above mean:", round(prob, 2))
print("4. Correlation between numerical features analyzed.")
print("5. Hypothesis test result p-value:", round(p_value, 4))
