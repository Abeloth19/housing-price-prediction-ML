# California Housing Price Prediction

A machine learning project that predicts house prices in California using Linear Regression.

## What it does

Give it information about a house area (income, location, age, etc.) and it predicts the house price.

**Example:**
- Input: Median income $60K, 25 years old, near San Francisco
- Output: Predicted price $285,000

## Results

- **Accuracy**: 66% of price variation explained (R² = 0.66)
- **Average Error**: $19,500 
- **Dataset**: 20,640 California houses with 8 features
- **Best Performance**: Medium-priced houses ($200K-$350K range)

## How to run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abeloth19/housing-price-prediction-ML.git
   cd "housing-price-prediction-ML"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv house_price_env
   house_price_env\Scripts\activate  # Windows
   # source house_price_env/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the model**
   ```bash
   python house_price_model.py
   ```

The script will automatically load data, train the model, and show results with visualizations.

## Key Features

- **Data Analysis**: Explores 20,640 California housing records
- **Data Preprocessing**: Handles outliers, feature scaling, train/test split
- **Model Training**: Linear Regression with cross-validation
- **Evaluation**: Multiple metrics and visualizations
- **Sample Predictions**: Tests model with example houses

## Tech Stack

- **Python 3.7+**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **matplotlib/seaborn** - Visualizations
- **numpy** - Numerical computations

## What you'll see

The model will show you:
- Data exploration charts
- Feature correlation analysis  
- Model performance metrics
- Prediction accuracy visualizations
- Sample house price predictions

## Sample Results

| House Type | Features | Predicted Price |
|------------|----------|----------------|
| Luxury LA | High income, new, coastal | $420,000 |
| Budget Central CA | Low income, old, inland | $180,000 |
| Suburban Bay Area | Medium income, mid-age | $310,000 |

## Model Performance

- **Training R²**: 0.67
- **Test R²**: 0.66  
- **Cross-validation**: 0.65 ± 0.02
- **Most Important Features**: Median Income, Latitude, Longitude

Built as a learning project to demonstrate machine learning fundamentals and data science workflow.
