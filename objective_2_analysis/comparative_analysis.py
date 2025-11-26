import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
DATA_FILES = {
    'OEM1': '../data/1.csv',
    'OEM2': '../data/2.csv',
    'OEM3': '../data/3.csv'
}
OUTPUT_DIR = 'results'
COLUMNS = [
    'DatasetID', 'CellID', 'CycleID', 'StepID', 'RecordID', 'Timestamp',
    'TestTime_min', 'Voltage_mV', 'Current_mA', 'Capacity_mAh', 
    'Energy_mWh', 'Power_W'
]

def load_all_data():
    dfs = []
    print("Loading data from all OEMs...")
    for oem, path in DATA_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, skiprows=1, names=COLUMNS)
            df['OEM'] = oem
            # Preprocessing: Sort is critical due to interleaved data
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
            df.sort_values(by=['CellID', 'Timestamp'], inplace=True)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
    
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def analyze_oems():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_all_data()
    if df is None: return

    # --- 1. Calculate Internal Resistance (DCIR) ---
    # Logic: R = dV / dI at Rest -> Discharge transitions
    print("Calculating Internal Resistance (DCIR)...")
    
    # Identify Discharge State (Current < -50mA)
    df['Is_Discharge'] = df['Current_mA'] < -50
    
    # Shift to find previous state (Rest)
    df['Prev_I'] = df.groupby(['OEM', 'CellID'])['Current_mA'].shift(1)
    df['Prev_V'] = df.groupby(['OEM', 'CellID'])['Voltage_mV'].shift(1)
    
    # Filter for Rest -> Discharge transition
    transitions = df[
        (df['Prev_I'].abs() < 20) &       # Previous was Rest (~0A)
        (df['Is_Discharge'])              # Current is Discharge
    ].copy()
    
    # Calculate dV and dI
    transitions['dV'] = (transitions['Prev_V'] - transitions['Voltage_mV']).abs() / 1000.0 # Volts
    transitions['dI'] = (transitions['Prev_I'] - transitions['Current_mA']).abs() / 1000.0 # Amps
    
    # Ohm's Law: R = V / I
    transitions['DCIR'] = transitions['dV'] / transitions['dI']
    
    # Clean DCIR (remove outliers/noise > 1.5 Ohm)
    transitions = transitions[(transitions['DCIR'] > 0) & (transitions['DCIR'] < 1.5)]
    
    # Aggregate DCIR per Cell (Median to be robust to outliers)
    cell_dcir = transitions.groupby(['OEM', 'CellID'])['DCIR'].median().reset_index()

    # --- 2. Calculate Max Capacity per Cell ---
    print("Calculating Max Capacity...")
    discharge_data = df[df['Is_Discharge']]
    cell_capacity = discharge_data.groupby(['OEM', 'CellID'])['Capacity_mAh'].max().reset_index()

    # Merge Metrics
    metrics = pd.merge(cell_capacity, cell_dcir, on=['OEM', 'CellID'])
    
    # --- 3. Generate Plots ---
    print("Generating plots...")
    
    # Plot 1: Capacity Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics, x='OEM', y='Capacity_mAh', palette='viridis')
    plt.title('Discharge Capacity Distribution by OEM')
    plt.ylabel('Capacity (mAh)')
    plt.savefig(os.path.join(OUTPUT_DIR, "capacity_comparison.png"))
    
    # Plot 2: DCIR Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics, x='OEM', y='DCIR', palette='magma')
    plt.title('Internal Resistance (DCIR) Distribution by OEM')
    plt.ylabel('DCIR (Ohms)')
    plt.savefig(os.path.join(OUTPUT_DIR, "dcir_comparison.png"))
    
    # Plot 3: Scatter Performance Map
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=metrics, x='DCIR', y='Capacity_mAh', hue='OEM', style='OEM', s=80, alpha=0.8)
    plt.title('Performance Map: Energy vs Power (Resistance)')
    plt.xlabel('Internal Resistance (Ohms) [Lower is Better]')
    plt.ylabel('Capacity (mAh) [Higher is Better]')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_map.png"))
    
    print("Analysis complete.")

if __name__ == "__main__":
    analyze_oems()
