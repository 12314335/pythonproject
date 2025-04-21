import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
df = pd.read_csv(r"12314335python/pav.csv")
print(df.head())
df.info()
df.describe()
print("Missing values:\n", df.isnull().sum())
print(df.isnull().sum().sum())




# Drop non-numeric columns
df = df.drop(columns=['State', 'District', 'Block', 'Village', 'Water Quality Classification'], errors='ignore')

# Box Plots for Key Features
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="WQI")
plt.title("Boxplot of House Prices")
plt.show()






# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()





# Distribution Plot for WQI
sns.histplot(df['WQI'], bins=30, kde=True)
plt.title("WQI Distribution")
plt.show()




selected_features = ['TDS']

# Drop rows with any missing values in selected features (if any)
pairplot_data = df[selected_features].dropna()

#Create pair plot of TDS
sns.pairplot(pairplot_data, diag_kind='kde', corner=True)
plt.suptitle("Pair Plot of water tds", y=1.02)
plt.show()


#bar plot
average_wqi = df['WQI'].mean()
print(f"Average WQI: {average_wqi:.2f}")
plt.bar(['Average WQI'], [average_wqi], color='orange')
plt.title('Average Water Quality Index')
plt.ylabel('WQI')
plt.show()
'''
#count plot on wql
sns.countplot(data=df, x='WQI')
plt.xticks(rotation=45)  # Rotate x labels if needed
plt.title("Count Plot of your_column")
plt.show()
'''
df = df.drop(columns=['State', 'District', 'Block', 'Village', 'Water Quality Classification'], errors='ignore')

# Extract WQI column and drop NA
wqi_data = df['WQI'].dropna()

# Parameters
mu_0 = 50  # Hypothesized population mean

# Z-test calculation
sample_mean = np.mean(wqi_data)
sample_std = np.std(wqi_data, ddof=1)
n = len(wqi_data)
z_score = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Z-score: {z_score:.3f}")
print(f"P-value: {p_value:.4f}")

# Plotting
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Standard Normal Distribution')
plt.axvline(z_score, color='red', linestyle='--', label=f'Z-score = {z_score:.2f}')
plt.axvline(-z_score, color='red', linestyle='--')
plt.fill_between(x, y, where=(x > abs(z_score)) | (x < -abs(z_score)), color='red', alpha=0.3, label='Rejection Region')
plt.title('Z-Test for WQI Mean')
plt.xlabel('Z')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

























