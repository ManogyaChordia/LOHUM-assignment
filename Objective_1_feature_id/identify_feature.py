import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
DATA_PATH = '../data/1.csv'  # Using OEM 1 as the sample
OUTPUT_DIR = 'results'
COLUMNS = [
    'DatasetID', 'CellID', 'CycleID', 'StepID', 'RecordID', 'Timestamp',
    'TestTime_min', 'Voltage_mV', 'Current_mA', 'Capacity_mAh', 
    'Feature_10_mWh', 'Power_W'
]

def analyze_feature_10():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, header=None, skiprows=1, names=COLUMNS)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Preprocessing: Sort by Timestamp to fix interleaved data
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df.sort_values(by=['CellID', 'Timestamp'], inplace=True)

    # 1. Theoretical Derivation
    print("Calculating theoretical energy...")
    
    # Power (W) = V * I. 
    # Energy (Wh) = Power * Time.
    
    # Calculate Instantaneous Power (W)
    # Voltage is mV -> V, Current is mA -> A
    power_w = (df['Voltage_mV'] / 1000.0) * (df['Current_mA'] / 1000.0).abs()
    
    # Calculate Time Delta (Hours) between records
    # We group by CellID to ensure we don't diff across different cells
    time_diff_h = df.groupby('CellID')['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600.0
    
    # 2. Correlation Check
    # Feature 10 is likely Cumulative Energy. 
    # To verify, we correlate Feature 10 against a proxy: Capacity * Voltage
    # Capacity (mAh) * Voltage (V) ~= Energy (mWh)
    # This avoids the complexity of resetting integrals per cycle step for verification.
    
    energy_proxy = df['Capacity_mAh'] * (df['Voltage_mV'] / 1000.0)
    
    correlation = df['Feature_10_mWh'].corr(energy_proxy)
    print(f"\nCorrelation between Feature 10 and Energy Proxy (Cap * V): {correlation:.6f}")

    # 3. Visualize
    plt.figure(figsize=(8, 6))
    # Sample 5000 points for speed and clarity
    sample = df.sample(5000, random_state=42)
    sample_proxy = sample['Capacity_mAh'] * (sample['Voltage_mV'] / 1000.0)
    
    sns.scatterplot(x=sample_proxy, y=sample['Feature_10_mWh'], alpha=0.5)
    
    # Add identity line (visual guide)
    max_val = max(sample_proxy.max(), sample['Feature_10_mWh'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Linear Trend')
    
    plt.xlabel('Calculated Energy Proxy (Capacity * Voltage) [mWh]')
    plt.ylabel('Feature 10 [mWh]')
    plt.title(f'Feature 10 Verification: Energy (R = {correlation:.4f})')
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(OUTPUT_DIR, "feature_10_correlation.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    analyze_feature_10()
