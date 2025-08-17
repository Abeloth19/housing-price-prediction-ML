# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the California Housing dataset
print("Loading California Housing Dataset...")
california = fetch_california_housing()

# Convert to pandas DataFrame for easier handling
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

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Create visualizations
print("\nCreating visualizations...")

# Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('California Housing Dataset - Initial Exploration', fontsize=16, fontweight='bold')

# 1. Price Distribution
axes[0, 0].hist(df['PRICE'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of House Prices')
axes[0, 0].set_xlabel('Price (hundreds of thousands $)')
axes[0, 0].set_ylabel('Frequency')

# 2. Income vs Price Scatter Plot
axes[0, 1].scatter(df['MedInc'], df['PRICE'], alpha=0.5, color='green', s=1)
axes[0, 1].set_title('Median Income vs House Price')
axes[0, 1].set_xlabel('Median Income')
axes[0, 1].set_ylabel('Price (hundreds of thousands $)')

# 3. House Age vs Price
axes[1, 0].scatter(df['HouseAge'], df['PRICE'], alpha=0.5, color='orange', s=1)
axes[1, 0].set_title('House Age vs Price')
axes[1, 0].set_xlabel('House Age (years)')
axes[1, 0].set_ylabel('Price (hundreds of thousands $)')

# 4. Average Rooms vs Price
axes[1, 1].scatter(df['AveRooms'], df['PRICE'], alpha=0.5, color='red', s=1)
axes[1, 1].set_title('Average Rooms vs Price')
axes[1, 1].set_xlabel('Average Rooms per Household')
axes[1, 1].set_ylabel('Price (hundreds of thousands $)')

plt.tight_layout()
plt.show()

# Correlation Matrix
correlation_matrix = df.corr()
print("\nCorrelation with PRICE:")
price_correlations = correlation_matrix['PRICE'].sort_values(ascending=False)
print(price_correlations)

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 2: DATA PREPROCESSING & PREPARATION
# ============================================================================

# Outlier Detection and Analysis
print("\nAnalyzing outliers...")

# Check for extreme outliers in key features
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Analyze outliers in key features
outlier_features = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'PRICE']
print("Outlier analysis:")
for feature in outlier_features:
    outliers, lower, upper = detect_outliers(df, feature)
    print(f"{feature}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Remove extreme outliers (optional - keeping moderate outliers for real-world data)
print(f"\nOriginal dataset size: {df.shape[0]}")

# Remove only the most extreme outliers
df_clean = df[
    (df['AveRooms'] < 20) &  # Remove houses with >20 average rooms
    (df['AveBedrms'] < 5) &  # Remove houses with >5 average bedrooms
    (df['AveOccup'] < 10)    # Remove houses with >10 average occupants
].copy()

print(f"After outlier removal: {df_clean.shape[0]}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} extreme outliers")

# Prepare features and target
X = df_clean.drop('PRICE', axis=1)  # Features
y = df_clean['PRICE']               # Target variable

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature Scaling/Normalization
print("\nApplying feature scaling...")
scaler = StandardScaler()

# Fit scaler on training data only (prevent data leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Feature scaling completed!")

# Verify the split maintains similar price distributions
print("\nVerifying data split quality:")
print(f"Training set price - Mean: ${y_train.mean():.3f}, Std: ${y_train.std():.3f}")
print(f"Testing set price - Mean: ${y_test.mean():.3f}, Std: ${y_test.std():.3f}")

# Create visualization to verify split quality
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Price distribution comparison
axes[0].hist(y_train, bins=30, alpha=0.7, label='Training Set', color='blue')
axes[0].hist(y_test, bins=30, alpha=0.7, label='Testing Set', color='red')
axes[0].set_title('Price Distribution: Train vs Test')
axes[0].set_xlabel('Price (hundreds of thousands $)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Feature scaling visualization (before vs after)
axes[1].boxplot([X_train['MedInc'], X_train['Population']], labels=['MedInc', 'Population'])
axes[1].set_title('Features Before Scaling')
axes[1].set_ylabel('Original Values')

axes[2].boxplot([X_train_scaled['MedInc'], X_train_scaled['Population']], labels=['MedInc', 'Population'])
axes[2].set_title('Features After Scaling')
axes[2].set_ylabel('Scaled Values')

plt.tight_layout()
plt.show()

# Display sample of scaled features
print("\nSample of scaled features:")
print("Before scaling:")
print(X_train.head(3))
print("\nAfter scaling:")
print(X_train_scaled.head(3))

print("\nData preprocessing completed!")
print("Ready for model training...")