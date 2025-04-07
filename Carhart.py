import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ---------------------------
# Global Configuration
# ---------------------------
MIN_SAMPLES_PER_SECTOR = 50
METRICS_COLUMNS = ['Sector', 'Samples', 'RMSE', 'MAE', 'MSE', 'R2', 'MAPE']

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    # 1. Load the data
    df = load_and_preprocess_data()

    # 2. Calculate Carhart four factor fomular
    df = compute_carhart_factors(df)

    # 3. Conduct industry regression analysis
    sector_metrics = []
    valid_sectors = identify_valid_sectors(df)

    # Processing industry model
    for sector in valid_sectors:
        sector_data = df[df['sector'] == sector].copy()

        if len(sector_data) < MIN_SAMPLES_PER_SECTOR:
            print(f"Skipping {sector}: Insufficient samples ({len(sector_data)})")
            continue

        print(f"\n{'=' * 40}\nProcessing {sector} ({len(sector_data)} samples)\n{'=' * 40}")

        metrics = process_sector(sector_data, sector)
        sector_metrics.append(metrics)

    # 4. Add global model
    global_data = df[df['sector'].isin(valid_sectors)].copy()
    print(f"\n{'=' * 40}\nProcessing Global Model ({len(global_data)} samples)\n{'=' * 40}")
    global_metrics = process_sector(global_data, "Global")
    sector_metrics.append(global_metrics)

    # 5. Interpretation of result
    analyze_performance(pd.DataFrame(sector_metrics, columns=METRICS_COLUMNS))

# ---------------------------
# Data Processing
# ---------------------------
def load_and_preprocess_data():
    """Load the data and preprocess it"""
    try:
        df = pd.read_excel('1.xlsx')
        print(f"Initial dataset size: {len(df):,}")
    except Exception as e:
        raise ValueError(f"Data loading error: {e}")

    # Renaming Column
    df = df.rename(columns={
        'horizon (days)': 'holding_period',
        'expected_return (yearly)': 'expected_return'
    })

    # Delete irrelevant column
    df = df.drop(columns=['company', 'date_BUY_fix', 'date_SELL_fix'], errors='ignore')

    # Processing class variable
    df['investment'] = LabelEncoder().fit_transform(df['investment'])

    # Calculate returns
    df = compute_returns(df)

    # Handle outliers
    df = filter_outliers(df)

    return df

def compute_returns(df):
    """ Calculated rate of return"""
    valid_prices = (df['price_BUY'] > 1e-6) & (df['price_SELL'] > 1e-6)
    df = df[valid_prices].copy()

    df['log_return'] = np.log(df['price_SELL'] / df['price_BUY'])
    df['annualized_return'] = df['log_return'] * (365 / df['holding_period'].clip(lower=1))
    df['real_return'] = df['annualized_return'] - df['inflation']
    return df

def filter_outliers(df):
    """ Removal of extremes"""
    lower, upper = df['real_return'].quantile([0.02, 0.98])
    print(f"Outlier filtering range: [{lower:.4f}, {upper:.4f}]")
    return df[(df['real_return'] > lower) & (df['real_return'] < upper)]

# ---------------------------
# Carhart Four-factor fomular calculation
# ---------------------------
def compute_carhart_factors(df):
    """ Calculate Carhart four factor fomular"""

    # Calculate Mkt-RF (Market Excess Return)
    df['Mkt_RF'] = df['expected_return'].mean() - df['inflation']

    # Calculate SMB (Size factor)
    median_pb = df['PB_ratio'].median()
    df['SMB'] = np.where(df['PB_ratio'] < median_pb, df['nominal_return'], -df['nominal_return'])

    # Calculate HML (Value factor)
    median_pe = df['PE_ratio'].median()
    df['HML'] = np.where(df['PE_ratio'] < median_pe, df['nominal_return'], -df['nominal_return'])

    # Calculating MOM (Momentum factor)
    median_mom = df['nominal_return'].median()
    df['MOM'] = np.where(df['nominal_return'] > median_mom, df['nominal_return'], -df['nominal_return'])

    return df

# ---------------------------
# Sector Processing
# ---------------------------
def identify_valid_sectors(df):
    """ Screen samples enough for the industry"""
    sector_counts = df['sector'].value_counts()
    return sector_counts[sector_counts >= MIN_SAMPLES_PER_SECTOR].index.tolist()

def process_sector(sector_df, sector_name):
    """ Industry regression analysis"""
    # Use the Carhart four factors
    X, y = sector_df[['Mkt_RF', 'SMB', 'HML', 'MOM']], sector_df['real_return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use OLS linear regression 
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluation Model 
    metrics = evaluate_model(model, X_test, y_test, sector_name, len(sector_df))

    # SHAP analysis
    explain_model_with_shap(model, X_test, sector_name)

    # Conservation of resources model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/model_{sector_name.replace(' ', '_')}.pkl")

    return metrics

def evaluate_model(model, X_test, y_test, sector_name, n_samples):
    """ Calculated regression index"""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    epsilon = 1e-6
    mape = 100 * np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + epsilon)))

    return [sector_name, n_samples, rmse, mae, mse, r2, mape]

def explain_model_with_shap(model, X_test, sector_name):
    """ Use SHAP interpretation model"""
    explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)

    # Save SHAP value
    pd.DataFrame(shap_values, columns=X_test.columns).to_csv(f"shap_values_{sector_name.replace(' ', '_')}.csv",
                                                                    index=False)

    # Generative visualization
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary - {sector_name}')
    plt.savefig(f'shap_summary_{sector_name.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

# ---------------------------
# Result Analysis
# ---------------------------
def analyze_performance(metrics_df):
    """ Analyze industry and global model results"""
    print("\nPerformance Comparison:")
    print(metrics_df.sort_values('RMSE'))

    # Separate global and industry results
    global_metrics = metrics_df[metrics_df['Sector'] == 'Global']
    sector_metrics = metrics_df[metrics_df['Sector'] != 'Global']

    # Visual contrast
    plt.figure(figsize=(15, 10))

    # Error index comparison
    plt.subplot(2, 1, 1)
    ax = sector_metrics.set_index('Sector')[['RMSE', 'MAE']].plot(kind='bar', alpha=0.8)
    plt.axhline(y=global_metrics['RMSE'].values[0], color='r', linestyle='--', label='Global RMSE')
    plt.axhline(y=global_metrics['MAE'].values[0], color='g', linestyle='--', label='Global MAE')
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Error Values')
    plt.legend()

    # Global model feature importance
    plt.subplot(2, 1, 2)
    try:
        shap_global = pd.read_csv('shap_values_Global.csv')
        (shap_global.abs().mean()
         .sort_values(ascending=False)
         .plot(kind='barh', title='Global Model Feature Importance (SHAP)'))
    except Exception as e:
        print(f"Error loading SHAP values: {e}")

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

# ---------------------------
# Execution
# ---------------------------
if __name__ == "__main__":
    main()
