import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lime
import lime.lime_tabular

# ---------------------------
# **Global Configuration**
# ---------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# ---------------------------
# **Data Loading & Preprocessing**
# ---------------------------
def load_and_preprocess():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_excel('1.xlsx')
        print(f"✅ Data loaded successfully, sample count: {len(df):,}")
    except Exception as e:
        print(f"❌ File loading failed: {e}")
        exit(1)

    df = df.rename(columns={
        'horizon (days)': 'holding_period',
        'expected_return (yearly)': 'expected_return'
    })

    # Validate required columns
    required_columns = ['price_BUY', 'price_SELL', 'sector']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        exit(1)

    # Remove irrelevant columns
    df = df.drop(columns=['company', 'date_BUY_fix', 'date_SELL_fix', 'investment'], errors='ignore')

    # Calculate returns
    df['log_return'] = np.log(df['price_SELL'] / df['price_BUY'])
    df['annualized_return'] = df['log_return'] * (365 / df['holding_period'].clip(lower=1))
    df['real_return'] = df['annualized_return'] - df['inflation']

    # Filter outliers
    lower, upper = df['real_return'].quantile([0.01, 0.99])
    df = df[(df['real_return'] > lower) & (df['real_return'] < upper)]

    return df.reset_index(drop=True)


# ---------------------------
# **Feature Engineering**
# ---------------------------
def generate_features(df):
    """Generate features"""
    base_features = [
        'PE_ratio', 'Volatility_Buy', 'ROE_ratio',
        'NetProfitMargin_ratio', 'current_ratio', 'PS_ratio',
        'holding_period'
    ]

    # Validate features
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        print(f"❌ Missing required features: {missing_features}")
        exit(1)

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(df[base_features])
    y = df['real_return'].values.astype(np.float32)

    return X, y, base_features, scaler


# ---------------------------
# **LIME Explanation**
# ---------------------------
def lime_explanation(model, scaler, df, feature_names, sample_ratio=0.1):
    """LIME explanation for overall model"""
    try:
        # Get standardized data
        X_all = scaler.transform(df[feature_names])

        # Prediction validation
        preds = model.predict(X_all)
        print(f"📈 Prediction stats - Mean: {np.mean(preds):.4f}, Std: {np.std(preds):.4f}, "
              f"Range: [{np.min(preds):.4f}, {np.max(preds):.4f}]")

        # Minimum 5% variation required
        if np.std(preds) < 1e-4:
            print(f"⚠️ Model shows minimal prediction variation, explanation aborted")
            return

        # Initialize explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_all,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=False,
            kernel_width=3,
            random_state=42
        )

        # Sampling logic
        num_samples = max(int(len(X_all) * sample_ratio), 50)
        sample_indices = np.random.choice(len(X_all), num_samples, replace=False)

        feature_importances = np.zeros(len(feature_names))
        valid_explanations = 0

        for idx in sample_indices:
            try:
                exp = explainer.explain_instance(
                    X_all[idx],
                    model.predict,
                    num_features=len(feature_names),
                    num_samples=500
                )
                weights = dict(exp.as_list())

                for i, feat in enumerate(feature_names):
                    feature_importances[i] += weights.get(feat, 0)

                valid_explanations += 1
            except Exception as e:
                print(f"⚠️ Explanation failed (sample {idx}): {str(e)}")

        if valid_explanations == 0:
            print("❌ No valid explanations generated")
            return

        feature_importances /= valid_explanations

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances, color='darkblue', alpha=0.7)
        plt.title(f"LIME Feature Importance - ALL\n(Based on {valid_explanations} samples)", fontsize=14)
        plt.xlabel("Importance Score")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"LIME_Overall.png", dpi=150)
        plt.close()
        print(f"✅ LIME explanation saved: LIME_Overall.png")

    except Exception as e:
        print(f"❌ Critical error in LIME explanation: {str(e)}")


# ---------------------------
# **Main Workflow**
# ---------------------------
def main():
    df = load_and_preprocess()
    X, y, feature_names, scaler = generate_features(df)

    # Train overall model
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8)
    model.fit(X, y)

    # Generate LIME explanation for overall model
    lime_explanation(model, scaler, df, feature_names)


if __name__ == "__main__":
    main()
