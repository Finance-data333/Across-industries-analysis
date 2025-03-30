import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 设置全局字体大小

# ---------------------------
# Data Preprocessing
# ---------------------------
def load_and_preprocess():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_excel('1.xlsx')
        print(f"Data loaded successfully, total samples: {len(df):,}")
    except Exception as e:
        print(f"Failed to load file: {e}")
        exit(1)

    # Standardizing column names
    df = df.rename(columns={
        'horizon (days)': 'holding_period',
        'expected_return (yearly)': 'expected_return'
    })

    # Validate required columns
    required_columns = ['price_BUY', 'price_SELL', 'sector']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        exit(1)

    # Drop irrelevant columns
    df = df.drop(columns=['company', 'date_BUY_fix', 'date_SELL_fix', 'investment'], errors='ignore')

    # Compute returns
    df['log_return'] = np.log(df['price_SELL'] / df['price_BUY'])
    df['annualized_return'] = df['log_return'] * (365 / df['holding_period'].clip(lower=1))
    df['real_return'] = df['annualized_return'] - df['inflation']

    # Filter outliers
    lower, upper = df['real_return'].quantile([0.01, 0.99])
    df = df[(df['real_return'] > lower) & (df['real_return'] < upper)]

    return df.reset_index(drop=True)


# ---------------------------
# Feature Engineering
# ---------------------------
def generate_features(df):
    """Generate the 7 main features."""
    base_features = [
        'PE_ratio', 'Volatility_Buy', 'ROE_ratio',
        'NetProfitMargin_ratio', 'current_ratio', 'PS_ratio',
        'holding_period'
    ]

    # Validate features
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        print(f"Missing essential features: {missing_features}")
        exit(1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[base_features])
    y = df['real_return'].values.astype(np.float32)

    return X, y, base_features, scaler


# ---------------------------
# Model Training (XGBoost version)
# ---------------------------
def train_model(X_train, y_train):
    """Train the XGBoost model with optimized parameters"""
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        tree_method='hist',
        enable_categorical=False
    )

    model.fit(X_train, y_train)
    return model


# ---------------------------
# SHAP Analysis by Sector (XGBoost adaptation)
# ---------------------------
def shap_by_sector(model, df, sector, scaler):
    """Perform SHAP analysis per sector."""
    features = ['PE_ratio', 'Volatility_Buy', 'ROE_ratio', 'NetProfitMargin_ratio',
                'current_ratio', 'PS_ratio', 'holding_period']

    # Normalize features
    X_sector = scaler.transform(df[features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sector)

    plt.figure(figsize=(12, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    shap.summary_plot(shap_values, features=X_sector, feature_names=features, show=False, plot_size=(12, 6))
    plt.title(f"{sector}", fontsize=14, fontweight='bold')
    plt.xlabel("SHAP Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.savefig(f"shap_sector_{sector}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sector {sector} SHAP analysis completed, image saved: shap_sector_{sector}.png")


# ---------------------------
# SHAP Analysis for All Sectors (XGBoost adaptation)
# ---------------------------
def shap_all_sectors(model, df, scaler):
    """Perform SHAP analysis for all sectors combined."""
    features = ['PE_ratio', 'Volatility_Buy', 'ROE_ratio', 'NetProfitMargin_ratio',
                'current_ratio', 'PS_ratio', 'holding_period']

    # Normalize features
    X_all = scaler.transform(df[features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)

    plt.figure(figsize=(12, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    shap.summary_plot(shap_values, features=X_all, feature_names=features, show=False, plot_size=(12, 6))
    plt.title("All", fontsize=14, fontweight='bold')
    plt.xlabel("SHAP Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.savefig("shap_all_sectors.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All sectors SHAP analysis completed, image saved: shap_all_sectors.png")


# ---------------------------
# Main Execution
# ---------------------------
def main():
    df = load_and_preprocess()
    sectors = df['sector'].unique()

    for sector in sectors:
        sector_df = df[df['sector'] == sector]
        if len(sector_df) < 10:
            print(f"Skipping {sector}, insufficient samples ({len(sector_df)})")
            continue

        print(f"\nProcessing sector: {sector} ({len(sector_df)} samples)")
        X, y, feature_names, scaler = generate_features(sector_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\nStarting model training...")
        model = train_model(X_train, y_train)

        joblib.dump(model, f'{sector}_xgb.pkl')
        joblib.dump(scaler, f'{sector}_scaler.pkl')
        print(f"\nModel saved successfully: {sector}_xgb.pkl")

        y_pred = model.predict(X_test)
        print("\nEvaluation Metrics:")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")

        print("\nPerforming SHAP analysis...")
        shap_by_sector(model, sector_df, sector, scaler)

    print("\nTraining model for all sectors combined...")
    X_all, y_all, feature_names_all, scaler_all = generate_features(df)
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    model_all = train_model(X_train_all, y_train_all)
    joblib.dump(model_all, 'all_sectors_xgb.pkl')
    joblib.dump(scaler_all, 'all_sectors_scaler.pkl')
    print(f"\nModel saved successfully: all_sectors_xgb.pkl")

    y_pred_all = model_all.predict(X_test_all)
    print("\nEvaluation Metrics for All Sectors:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_all, y_pred_all)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_all, y_pred_all):.4f}")
    print(f"R²: {r2_score(y_test_all, y_pred_all):.4f}")

    print("\nPerforming SHAP analysis for all sectors combined...")
    shap_all_sectors(model_all, df, scaler_all)


if __name__ == "__main__":
    main()