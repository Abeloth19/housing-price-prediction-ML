
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


print("\nAnalyzing outliers...")


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


outlier_features = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'PRICE']
print("Outlier analysis:")
for feature in outlier_features:
    outliers, lower, upper = detect_outliers(df, feature)
    print(f"{feature}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")


print(f"\nOriginal dataset size: {df.shape[0]}")


df_clean = df[
    (df['AveRooms'] < 20) & 
    (df['AveBedrms'] < 5) &  
    (df['AveOccup'] < 10)    
].copy()

print(f"After outlier removal: {df_clean.shape[0]}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} extreme outliers")


X = df_clean.drop('PRICE', axis=1)  
y = df_clean['PRICE']              

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")


print("\nApplying feature scaling...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Feature scaling completed!")


print("\nVerifying data split quality:")
print(f"Training set price - Mean: ${y_train.mean():.3f}, Std: ${y_train.std():.3f}")
print(f"Testing set price - Mean: ${y_test.mean():.3f}, Std: ${y_test.std():.3f}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].hist(y_train, bins=30, alpha=0.7, label='Training Set', color='blue')
axes[0].hist(y_test, bins=30, alpha=0.7, label='Testing Set', color='red')
axes[0].set_title('Price Distribution: Train vs Test')
axes[0].set_xlabel('Price (hundreds of thousands $)')
axes[0].set_ylabel('Frequency')
axes[0].legend()


axes[1].boxplot([X_train['MedInc'], X_train['Population']], labels=['MedInc', 'Population'])
axes[1].set_title('Features Before Scaling')
axes[1].set_ylabel('Original Values')

axes[2].boxplot([X_train_scaled['MedInc'], X_train_scaled['Population']], labels=['MedInc', 'Population'])
axes[2].set_title('Features After Scaling')
axes[2].set_ylabel('Scaled Values')

plt.tight_layout()
plt.show()


print("\nSample of scaled features:")
print("Before scaling:")
print(X_train.head(3))
print("\nAfter scaling:")
print(X_train_scaled.head(3))

print("\nData preprocessing completed!")
print("Ready for model training...")


print("\nTraining Linear Regression model...")
model = LinearRegression()


model.fit(X_train_scaled, y_train)
print("Model training completed!")


y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\nPredictions generated for training and test sets.")


train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))


print("\nModel Performance Metrics:")
print("=" * 40)
print(f"Training Set:")
print(f"  R² Score: {train_r2:.4f} ({train_r2*100:.1f}% variance explained)")
print(f"  MAE: ${train_mae:.3f} (hundreds of thousands)")
print(f"  RMSE: ${train_rmse:.3f} (hundreds of thousands)")

print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"  MAE: ${test_mae:.3f} (hundreds of thousands)")
print(f"  RMSE: ${test_rmse:.3f} (hundreds of thousands)")


print(f"\nIn actual dollars:")
print(f"  Test MAE: ${test_mae*100000:,.0f}")
print(f"  Test RMSE: ${test_rmse*100000:,.0f}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold', y=0.98)


axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='blue', s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Prices')
axes[0, 0].set_ylabel('Predicted Prices')
axes[0, 0].set_title(f'Predicted vs Actual (R² = {test_r2:.3f})', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)


residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, color='green', s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Prices')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals Plot', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)


axes[1, 0].barh(feature_importance['Feature'], feature_importance['Abs_Coefficient'])
axes[1, 0].set_xlabel('Absolute Coefficient Value')
axes[1, 0].set_title('Feature Importance', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution', fontsize=12)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].grid(True, alpha=0.3)


plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
plt.show()


print("\nModel Interpretation:")
print("=" * 40)
print(f"Intercept: ${model.intercept_:.3f} (hundreds of thousands)")
print("\nTop 3 most important features:")
for i, row in feature_importance.head(3).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: {direction} price by ${abs(row['Coefficient']):.3f} per unit change")


print("\nSample Predictions (first 5 test examples):")
print("=" * 50)
sample_predictions = pd.DataFrame({
    'Actual_Price': y_test.iloc[:5].values,
    'Predicted_Price': y_test_pred[:5],
    'Error': (y_test.iloc[:5].values - y_test_pred[:5]),
    'Error_Percentage': ((y_test.iloc[:5].values - y_test_pred[:5]) / y_test.iloc[:5].values * 100)
})

for i, row in sample_predictions.iterrows():
    print(f"House {i+1}:")
    print(f"  Actual: ${row['Actual_Price']*100000:,.0f}")
    print(f"  Predicted: ${row['Predicted_Price']*100000:,.0f}")
    print(f"  Error: ${row['Error']*100000:,.0f} ({row['Error_Percentage']:.1f}%)")
    print()


print("Model Training Summary:")
print("=" * 40)
if test_r2 >= 0.75:
    performance = "Excellent"
elif test_r2 >= 0.65:
    performance = "Good"
elif test_r2 >= 0.50:
    performance = "Fair"
else:
    performance = "Needs Improvement"

print(f"Overall Performance: {performance}")
print(f"The model explains {test_r2*100:.1f}% of house price variation")
print(f"Average prediction error: ${test_mae*100000:,.0f}")
print("Model is ready for making predictions!")