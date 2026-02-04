# ===============================
# TASK 10: EDA + OUTLIER DETECTION
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
file_path = r"D:\Internship\students_performance.csv"   # CHANGE PATH
df = pd.read_csv(file_path)

# 2️⃣ Basic Information
print("SHAPE:", df.shape)
print("\nINFO:")
print(df.info())
print("\nFIRST 5 ROWS:")
print(df.head())

# 3️⃣ Descriptive Statistics
print("\nDESCRIBE:")
print(df.describe())

# 4️⃣ Missing Values Percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\nMISSING VALUES (%):")
print(missing_percent)

# 5️⃣ Histogram and Boxplot
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(f"Histogram of {col}")
    plt.show()

    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.show()

# 6️⃣ IQR Outlier Detection
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Outlier Flag
outlier_flag = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

df["Outlier_Flag"] = outlier_flag
print("\nTotal Outliers:", df["Outlier_Flag"].sum())

# 7️⃣ Handle Outliers (Capping)
for col in numeric_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("\nOutliers capped using IQR method")

# 8️⃣ Correlation Matrix
corr = df[numeric_cols].corr()
print("\nCORRELATION MATRIX:")
print(corr)

# Plot Correlation Heatmap
plt.figure(figsize=(10,8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix Heatmap")
plt.show()

# 9️⃣ Export Cleaned Dataset
output_path = r"D:\Internship\cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print("\nCleaned dataset saved to:", output_path)

# 10️⃣ Save EDA Findings
with open(r"D:\Internship\eda_findings.txt", "w") as f:
    f.write("EDA Findings Summary\n")
    f.write(str(df.describe()))
    f.write("\n\nCorrelation Matrix:\n")
    f.write(str(corr))

print("EDA findings saved as eda_findings.txt")
