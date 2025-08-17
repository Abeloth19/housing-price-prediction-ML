import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


plt.style.use('default')
sns.set_palette("husl")


print("Loading California Housing Dataset...")
california = fetch_california_housing()


df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(california.feature_names)}")
print("\nDataset Description:")
print("- MedInc: Median income in block group")
print("- HouseAge: Median house age in block group") 
print("- AveRooms: Average number of rooms per household")
print("- AveBedrms: Average number of bedrooms per household")
print("- Population: Block group population")
print("- AveOccup: Average number of household members")
print("- Latitude: Block group latitude")
print("- Longitude: Block group longitude")
print("- PRICE: Median house value in hundreds of thousands of dollars")

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())


print("\nMissing values per column:")
print(df.isnull().sum())


print("\nCreating visualizations...")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('California Housing Dataset - Initial Exploration', fontsize=16, fontweight='bold')


axes[0, 0].hist(df['PRICE'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of House Prices')
axes[0, 0].set_xlabel('Price (hundreds of thousands $)')
axes[0, 0].set_ylabel('Frequency')


axes[0, 1].scatter(df['MedInc'], df['PRICE'], alpha=0.5, color='green', s=1)
axes[0, 1].set_title('Median Income vs House Price')
axes[0, 1].set_xlabel('Median Income')
axes[0, 1].set_ylabel('Price (hundreds of thousands $)')


axes[1, 0].scatter(df['HouseAge'], df['PRICE'], alpha=0.5, color='orange', s=1)
axes[1, 0].set_title('House Age vs Price')
axes[1, 0].set_xlabel('House Age (years)')
axes[1, 0].set_ylabel('Price (hundreds of thousands $)')


axes[1, 1].scatter(df['AveRooms'], df['PRICE'], alpha=0.5, color='red', s=1)
axes[1, 1].set_title('Average Rooms vs Price')
axes[1, 1].set_xlabel('Average Rooms per Household')
axes[1, 1].set_ylabel('Price (hundreds of thousands $)')

plt.tight_layout()
plt.show()


correlation_matrix = df.corr()
print("\nCorrelation with PRICE:")
price_correlations = correlation_matrix['PRICE'].sort_values(ascending=False)
print(price_correlations)


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()