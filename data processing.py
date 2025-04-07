import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Set the global font size
# ---------------------------
# Data Processing
# ---------------------------
def load_and_preprocess():
    """Complete data processing pipeline"""
    try:
        # File existence check
        if not os.path.exists('1.xlsx'):
            raise FileNotFoundError("Error: 1.xlsx file not found")

        # Read data with date parsing
        df = pd.read_excel(
            '1.xlsx',
            parse_dates=['date_BUY_fix', 'date_SELL_fix'],
            engine='openpyxl'
        )
        print(f"Data loaded successfully. Initial samples: {len(df):,}")

        # Validate required columns
        required_cols = ['price_BUY', 'price_SELL', 'date_BUY_fix', 'date_SELL_fix', 'inflation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {', '.join(missing)}")

        # Calculate holding period
        df['holding_period'] = (df['date_SELL_fix'] - df['date_BUY_fix']).dt.days + 1
        df = df[df['holding_period'] > 0]  # Filter invalid durations

        # Remove unnecessary columns
        df = df.drop(columns=['company', 'expected_return (yearly)'], errors='ignore')

        # Label encoding
        df['investment'] = LabelEncoder().fit_transform(df['investment'])

        # Calculate returns
        df = calculate_returns(df)

        # Outlier filtering
        df = filter_outliers(df)

        print(f"Final valid samples: {len(df):,}")
        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        exit(1)


def calculate_returns(df):
    """Return calculation logic"""
    # Price validation
    valid_prices = (df['price_BUY'] > 1e-6) & (df['price_SELL'] > 1e-6)
    df = df[valid_prices].copy()

    # Core calculations
    df['log_return'] = np.log(df['price_SELL'] / df['price_BUY'])
    df['annualized_return'] = df['log_return'] * (365 / df['holding_period'])
    df['real_return'] = df['annualized_return'] - df['inflation']

    return df


def filter_outliers(df):
    """Outlier filtering"""
    lower, upper = df['real_return'].quantile([0.01, 0.99])
    print(f"Outlier filter range: [{lower:.4f}, {upper:.4f}]")
    return df[(df['real_return'] > lower) & (df['real_return'] < upper)]


# ---------------------------
# Professional Visualization
# ---------------------------
def visualize_returns(df):
    """Return distribution visualization"""
    plt.figure(figsize=(12, 6))

    # Dual-axis system
    ax = plt.gca()
    ax2 = ax.twinx()

    # Histogram configuration
    hist = sns.histplot(
        df['real_return'],
        bins=40,
        kde=False,
        color='#4C72B0',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.7,
        ax=ax
    )

    # KDE configuration
    kde = sns.kdeplot(
        df['real_return'],
        color='#C44E52',
        linewidth=2.5,
        ax=ax2
    )

    # Statistical metrics
    stats = {
        'mean': df['real_return'].mean(),
        'median': df['real_return'].median(),
        'std': df['real_return'].std(),
        'skew': df['real_return'].skew(),
        'kurtosis': df['real_return'].kurtosis()
    }

    # Reference lines
    lines = [
        (stats['mean'], '#55A868', '--', f"Mean: {stats['mean']:.2f}"),
        (stats['median'], '#DD8452', ':', f"Median: {stats['median']:.2f}"),
        (stats['mean'] - stats['std'], '#4C8BB8', '-.', "μ-σ"),
        (stats['mean'] + stats['std'], '#4C8BB8', '-.', "μ+σ")
    ]

    # Draw reference lines
    for value, color, ls, label in lines:
        ax.axvline(
            x=value,
            color=color,
            linestyle=ls,
            linewidth=1.8,
            label=label,
            alpha=0.9
        )

    # Legend configuration
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.legend(
        [f"Skewness: {stats['skew']:.2f}\nKurtosis: {stats['kurtosis']:.2f}"],
        loc='upper right'
    )

    # Labels and titles
    ax.set_title(
        f"Real Return Distribution Analysis\n(Samples: {len(df):,} | Std Dev: {stats['std']:.2f})",
        fontsize=14,
        pad=20
    )
    ax.set_xlabel("Real Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)

    # Grid styling
    ax.grid(alpha=0.3, linestyle='--')
    ax2.grid(visible=False)

    # Save and display
    plt.tight_layout()
    plt.savefig(
        'return_analysis.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,  # 设置为False以避免透明背景
        facecolor='white'   # 设置背景颜色为白色
    )
    print("Visualization saved to return_analysis.png")
    plt.show()


# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Financial Return Analysis System v3.1")
    print("=" * 50)

    # Execute pipeline
    df = load_and_preprocess()

    # Data preview
    print("\nProcessed data sample:")
    print(df[['date_BUY_fix', 'date_SELL_fix', 'holding_period', 'real_return']].head(3))

    # Visualization
    visualize_returns(df)
