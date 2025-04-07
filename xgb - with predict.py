import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Set the global font size
# ---------------------------
# Global Configuration
# ---------------------------
MIN_SAMPLES_PER_SECTOR = 50
METRICS_COLUMNS = ['Sector', 'Samples', 'RMSE', 'MAE', 'MSE', 'R2', 'MAPE']


# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    # Data loading and preprocessing
    df = load_and_preprocess_data()

    # Sector analysis pipeline
    sector_metrics = []

    # Get qualified sectors
    valid_sectors = identify_valid_sectors(df)

    # Process each sector
    for sector in valid_sectors:
        sector_data = df[df['sector'] == sector].copy()

        if len(sector_data) < MIN_SAMPLES_PER_SECTOR:
            print(f"Skipping {sector}: Insufficient samples ({len(sector_data)})")
            continue

        print(f"\n{'=' * 40}\nProcessing {sector} ({len(sector_data)} samples)\n{'=' * 40}")

        metrics = process_sector(sector_data, sector)
        sector_metrics.append(metrics)

    # Analyze and visualize results
    analyze_performance(pd.DataFrame(sector_metrics, columns=METRICS_COLUMNS))

    # Process all sectors combined
    print("\n{'=' * 40}\nProcessing All Sectors Combined\n{'=' * 40}")
    process_all_sectors(df)


# ---------------------------
# Data Processing
# ---------------------------
def load_and_preprocess_data():
    """Main data processing pipeline"""
    try:
        df = pd.read_excel('1.xlsx')
        print(f"Initial dataset size: {len(df):,}")
    except Exception as e:
        raise ValueError(f"Data loading error: {e}")

    # Column standardization
    df = df.rename(columns={
        'horizon (days)': 'holding_period',
        'expected_return (yearly)': 'expected_return'
    })

    # Remove non-feature columns
    df = df.drop(columns=['company', 'date_BUY_fix', 'date_SELL_fix'], errors='ignore')

    # Encode categorical features
    df['investment'] = LabelEncoder().fit_transform(df['investment'])

    # Calculate returns
    df = compute_returns(df)

    # Handle outliers
    df = filter_outliers(df)

    return df


def compute_returns(df):
    """Calculate financial returns with validation"""
    valid_prices = (df['price_BUY'] > 1e-6) & (df['price_SELL'] > 1e-6)
    df = df[valid_prices].copy()

    df['log_return'] = np.log(df['price_SELL'] / df['price_BUY'])
    df['annualized_return'] = df['log_return'] * (365 / df['holding_period'].clip(lower=1))
    df['real_return'] = df['annualized_return'] - df['inflation']
    return df


def filter_outliers(df):
    """Remove extreme values using quantile-based filtering"""
    lower, upper = df['real_return'].quantile([0.02, 0.98])
    print(f"Outlier filtering range: [{lower:.4f}, {upper:.4f}]")
    return df[(df['real_return'] > lower) & (df['real_return'] < upper)]


# ---------------------------
# Sector Processing
# ---------------------------
def identify_valid_sectors(df):
    """Identify sectors with sufficient samples"""
    sector_counts = df['sector'].value_counts()
    return sector_counts[sector_counts >= MIN_SAMPLES_PER_SECTOR].index.tolist()


def process_sector(sector_df, sector_name):
    """Full processing pipeline for a single sector"""
    # Feature engineering
    X, y, preprocessor = prepare_features(sector_df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = train_sector_model(X_train, y_train, X_test, y_test)

    # Model evaluation
    metrics = evaluate_model(model, X_test, y_test, sector_name, len(sector_df))

    # Generate prediction plot
    generate_prediction_plot(y_test, model.predict(X_test), sector_name, metrics)

    # Save artifacts
    save_model_artifacts(model, preprocessor, sector_name)

    return metrics


def prepare_features(df):
    """Feature engineering pipeline for sector data"""
    numeric_features = [
        'PE_ratio', 'Volatility_Buy', 'ROE_ratio',
        'NetProfitMargin_ratio', 'current_ratio', 'PS_ratio',
        'investment', 'holding_period'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features)
        ],
        remainder='drop'
    )

    X = preprocessor.fit_transform(df)
    y = df['real_return'].values

    return X, y, preprocessor


def train_sector_model(X_train, y_train, X_val, y_val):
    """Train optimized XGBoost model for sector"""
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0
    )

    return model


def evaluate_model(model, X_test, y_test, sector_name, n_samples):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Handle division protection for MAPE
    epsilon = 1e-6
    mape = 100 * np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + epsilon)))

    return [sector_name, n_samples, rmse, mae, mse, r2, mape]


def generate_prediction_plot(y_test, y_pred, sector_name, metrics):
    """Generate prediction vs actual comparison plot using scatter plot"""
    # Null data check
    if len(y_test) == 0 or len(y_pred) == 0:
        print(f"Warning：{sector_name} Test data is empty, skip drawing")
        return

    plt.figure(figsize=(10, 8), dpi=300)

    try:
        # Plot scatter plots
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, s=80)

        # Add ideal prediction line（y = x）
        max_val = max(np.max(y_test), np.max(y_pred))
        min_val = min(np.min(y_test), np.min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

        # Add annotation
        textstr = '\n'.join((
            f'RMSE: {metrics[2]:.4f}',
            f'MAE: {metrics[3]:.4f}',
            f'R²: {metrics[5]:.4f}',
            f'MAPE: {metrics[6]:.2f}%'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                       fontsize=12, verticalalignment='top', bbox=props)

        # Format Settings
        plt.title(f'{sector_name}', fontsize=14, pad=20)
        plt.xlabel('Actual Real Return', fontsize=12, labelpad=10)
        plt.ylabel('Predicted Real Return', fontsize=12, labelpad=10)
        plt.legend(frameon=True, loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Create directory
        plot_dir = Path("prediction_plots")
        try:
            plot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Directory creation failure: {str(e)}")
            return

        # File name processing
        import re
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", sector_name)

        # Save image
        try:
            plt.savefig(plot_dir / f"{safe_name}_scatter_predictions.png",
                        bbox_inches='tight', pad_inches=0.3)
            print(f"Successful preservation {sector_name} scatter diagram")
        except Exception as e:
            print(f"Image saving failure: {str(e)}")

    except Exception as e:
        print(f"An error occurred during drawing: {str(e)}")
    finally:
        plt.close()  # Ensure resource release 


def save_model_artifacts(model, preprocessor, sector_name):
    """Save trained model and preprocessing pipeline"""
    output_dir = Path("sector_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_sector_name = sector_name.replace("/", "_").replace(":", "-")
    output_path = output_dir / f"{safe_sector_name}_pipeline.pkl"

    joblib.dump(
        {'model': model, 'preprocessor': preprocessor},
        output_path
    )
    print(f"Model saved to: {output_path.absolute()}")


# ---------------------------
# All Sectors Processing
# ---------------------------
def process_all_sectors(df):
    """Process all sectors combined"""
    # Feature engineering
    X, y, preprocessor = prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = train_sector_model(X_train, y_train, X_test, y_test)

    # Model evaluation
    metrics = evaluate_model(model, X_test, y_test, "All Sectors", len(df))

    # Generate prediction plot
    generate_prediction_plot(y_test, model.predict(X_test), "All Sectors", metrics)

    # Save artifacts
    save_model_artifacts(model, preprocessor, "all_sectors")


# ---------------------------
# Result Analysis
# ---------------------------
def analyze_performance(metrics_df):
    """Performance analysis and visualization"""
    print("\nSector Performance Analysis:")
    print(metrics_df.sort_values('RMSE'))

    # Error metrics comparison
    plt.figure(figsize=(15, 8), dpi=300)
    metrics_df.set_index('Sector')[['RMSE', 'MAE']].plot(kind='bar', alpha=0.8)
    plt.title('Cross-sector Error Metrics Comparison', fontsize=14)
    plt.ylabel('Error Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("sector_performance.png", dpi=300)
    plt.close()


# ---------------------------
# Execution
# ---------------------------
if __name__ == "__main__":
    main()
