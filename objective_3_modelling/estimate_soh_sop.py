import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os

# --- CONFIG ---
DATA_PATH = '../data/1.csv' # Modeling for OEM 1
OUTPUT_DIR = 'results'

# Rated values derived from global dataset analysis
RATED_CAPACITY = 4800.0 # mAh
RATED_POWER = 26.0      # Watts (Derived from OEM 2 peak performance)

COLUMNS = [
    'DatasetID', 'CellID', 'CycleID', 'StepID', 'RecordID', 'Timestamp',
    'TestTime_min', 'Voltage_mV', 'Current_mA', 'Capacity_mAh', 
    'Energy_mWh', 'Power_W'
]

def train_and_evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data for OEM 1...")
    try:
        df = pd.read_csv(DATA_PATH, header=None, skiprows=1, names=COLUMNS)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # Preprocessing
    df['Current_mA'] = pd.to_numeric(df['Current_mA'], errors='coerce')
    df['Voltage_mV'] = pd.to_numeric(df['Voltage_mV'], errors='coerce')
    df['Power_W'] = pd.to_numeric(df['Power_W'], errors='coerce')
    
    # Define States
    df['State'] = 'Rest'
    df.loc[df['Current_mA'] > 50, 'State'] = 'Charge'
    df.loc[df['Current_mA'] < -50, 'State'] = 'Discharge'

    # --- Feature Engineering (From CHARGE Cycle) ---
    print("Generating Features from Charge Cycle...")
    charge_data = df[df['State'] == 'Charge']
    
    # Aggregating charge statistics per cell/cycle
    features = charge_data.groupby(['CellID', 'CycleID']).agg({
        'Voltage_mV': ['mean', 'std', 'min', 'max'],
        'Current_mA': 'mean',
        'Power_W': 'mean',
        'TestTime_min': 'max' # Charge duration
    }).reset_index()
    
    # Flatten Hierarchical Columns
    features.columns = ['CellID', 'CycleID', 'Chg_V_Mean', 'Chg_V_Std', 'Chg_V_Min', 'Chg_V_Max', 'Chg_I_Mean', 'Chg_P_Mean', 'Chg_Time']

    # --- Target Generation (From DISCHARGE Cycle) ---
    print("Generating Targets from Discharge Cycle...")
    discharge_data = df[df['State'] == 'Discharge']
    
    targets = discharge_data.groupby(['CellID', 'CycleID']).agg({
        'Capacity_mAh': 'max',
        'Power_W': 'max' # Peak Power
    }).reset_index()
    targets.columns = ['CellID', 'CycleID', 'Discharge_Capacity', 'Peak_Power']

    # Merge Features and Targets
    model_df = pd.merge(features, targets, on=['CellID', 'CycleID'])
    
    # Calculate Targets: SOH and SOP
    model_df['SOH'] = model_df['Discharge_Capacity'] / RATED_CAPACITY
    model_df['SOP'] = model_df['Peak_Power'] / RATED_POWER

    # --- Model Training ---
    # Input Features
    X = model_df[['Chg_V_Mean', 'Chg_V_Std', 'Chg_V_Min', 'Chg_V_Max', 'Chg_I_Mean', 'Chg_P_Mean', 'Chg_Time']]
    # Targets
    y_soh = model_df['SOH']
    y_sop = model_df['SOP']

    rf_soh = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_sop = RandomForestRegressor(n_estimators=100, random_state=42)

    # Cross Validation (5-Fold)
    print("\nEvaluating SOH Model (5-Fold CV)...")
    scores_soh = cross_val_score(rf_soh, X, y_soh, cv=5, scoring='neg_mean_absolute_percentage_error')
    print(f"SOH MAPE: {-scores_soh.mean()*100:.4f}%")

    print("\nEvaluating SOP Model (5-Fold CV)...")
    scores_sop = cross_val_score(rf_sop, X, y_sop, cv=5, scoring='neg_mean_absolute_percentage_error')
    print(f"SOP MAPE: {-scores_sop.mean()*100:.4f}%")

    # --- Plotting Results ---
    rf_soh.fit(X, y_soh)
    preds_soh = rf_soh.predict(X)
    
    rf_sop.fit(X, y_sop)
    preds_sop = rf_sop.predict(X)
    
    # Plot SOH
    plt.figure(figsize=(8, 6))
    plt.scatter(y_soh, preds_soh, color='blue', alpha=0.6)
    plt.plot([y_soh.min(), y_soh.max()], [y_soh.min(), y_soh.max()], 'r--', label='Perfect Prediction')
    plt.title('SOH Estimation: Actual vs Predicted')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_soh_results.png"))
    
    # Plot SOP
    plt.figure(figsize=(8, 6))
    plt.scatter(y_sop, preds_sop, color='green', alpha=0.6)
    plt.plot([y_sop.min(), y_sop.max()], [y_sop.min(), y_sop.max()], 'r--', label='Perfect Prediction')
    plt.title('SOP Estimation: Actual vs Predicted')
    plt.xlabel('Actual SOP')
    plt.ylabel('Predicted SOP')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_sop_results.png"))
    
    print("Result plots saved.")

if __name__ == "__main__":
    train_and_evaluate()
