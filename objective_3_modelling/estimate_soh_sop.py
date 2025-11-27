import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error
)
import os

# --- CONFIG ---
DATA_PATH = '../data/1.csv'  # Modeling for OEM 1
OUTPUT_DIR = 'results'

# Rated values derived from global dataset analysis
RATED_CAPACITY = 4800.0  # mAh
RATED_POWER = 26.0       # Watts (Derived from OEM 2 peak performance)

COLUMNS = [
    'DatasetID', 'CellID', 'CycleID', 'StepID', 'RecordID', 'Timestamp',
    'TestTime_min', 'Voltage_mV', 'Current_mA', 'Capacity_mAh',
    'Energy_mWh', 'Power_W'
]


def compute_metrics(y_true, y_pred):
    """
    Returns (mape_fraction, mae, rmse)
    mape_fraction is e.g. 0.017 -> 1.7%
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, mae, rmse


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
        'TestTime_min': 'max'  # Charge duration
    }).reset_index()

    # Flatten Hierarchical Columns
    features.columns = [
        'CellID', 'CycleID',
        'Chg_V_Mean', 'Chg_V_Std', 'Chg_V_Min', 'Chg_V_Max',
        'Chg_I_Mean', 'Chg_P_Mean', 'Chg_Time'
    ]

    # --- Target Generation (From DISCHARGE Cycle) ---
    print("Generating Targets from Discharge Cycle...")
    discharge_data = df[df['State'] == 'Discharge']

    targets = discharge_data.groupby(['CellID', 'CycleID']).agg({
        'Capacity_mAh': 'max',
        'Power_W': 'max'  # Peak Power
    }).reset_index()
    targets.columns = ['CellID', 'CycleID', 'Discharge_Capacity', 'Peak_Power']

    # Merge Features and Targets
    model_df = pd.merge(features, targets, on=['CellID', 'CycleID'])

    if model_df.empty:
        print("No merged feature/target rows found. Check your CSV or state thresholds.")
        return

    # Calculate Targets: SOH and SOP
    model_df['SOH'] = model_df['Discharge_Capacity'] / RATED_CAPACITY
    model_df['SOP'] = model_df['Peak_Power'] / RATED_POWER

    # --- Model Training ---
    # Input Features
    X = model_df[['Chg_V_Mean', 'Chg_V_Std', 'Chg_V_Min', 'Chg_V_Max',
                  'Chg_I_Mean', 'Chg_P_Mean', 'Chg_Time']]
    # Targets
    y_soh = model_df['SOH'].values
    y_sop = model_df['SOP'].values

    rf_soh = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_sop = RandomForestRegressor(n_estimators=100, random_state=42)

    # --- 5-Fold Cross-Validation (out-of-fold predictions) ---
    cv = 5
    if len(model_df) < cv:
        print(f"Not enough samples ({len(model_df)}) for cv={cv}. Reduce cv or increase samples.")
        return

    print(f"\nPerforming {cv}-Fold CV and computing MAPE, MAE, RMSE (out-of-fold predictions)...")

    # Obtain out-of-fold predictions for both targets
    preds_soh_oof = cross_val_predict(rf_soh, X, y_soh, cv=cv, n_jobs=-1)
    preds_sop_oof = cross_val_predict(rf_sop, X, y_sop, cv=cv, n_jobs=-1)

    # Compute metrics
    soh_mape, soh_mae, soh_rmse = compute_metrics(y_soh, preds_soh_oof)
    sop_mape, sop_mae, sop_rmse = compute_metrics(y_sop, preds_sop_oof)

    # Print nicely
    print("\n--- 5-Fold CV Results (out-of-fold) ---")
    print(f"SOH MAPE: {soh_mape * 100:.4f}%")
    print(f"SOH MAE:  {soh_mae:.6f}")
    print(f"SOH RMSE: {soh_rmse:.6f}\n")

    print(f"SOP MAPE: {sop_mape * 100:.4f}%")
    print(f"SOP MAE:  {sop_mae:.6f}")
    print(f"SOP RMSE: {sop_rmse:.6f}")

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        'cv': cv,
        'SOH_MAPE_%': soh_mape * 100,
        'SOH_MAE': soh_mae,
        'SOH_RMSE': soh_rmse,
        'SOP_MAPE_%': sop_mape * 100,
        'SOP_MAE': sop_mae,
        'SOP_RMSE': sop_rmse
    }])
    metrics_summary.to_csv(os.path.join(OUTPUT_DIR, 'cv_metrics_summary_5fold.csv'), index=False)
    print(f"\nSaved CV metrics summary to: {os.path.join(OUTPUT_DIR, 'cv_metrics_summary_5fold.csv')}")

    # --- Fit on full data and save diagnostic plots (same as before) ---
    rf_soh.fit(X, y_soh)
    preds_soh_full = rf_soh.predict(X)

    rf_sop.fit(X, y_sop)
    preds_sop_full = rf_sop.predict(X)

    # Plot SOH
    plt.figure(figsize=(8, 6))
    plt.scatter(y_soh, preds_soh_full, color='blue', alpha=0.6)
    plt.plot([y_soh.min(), y_soh.max()], [y_soh.min(), y_soh.max()], 'r--', label='Perfect Prediction')
    plt.title('SOH Estimation: Actual vs Predicted (train fit)')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_soh_results.png"))
    plt.close()

    # Plot SOP
    plt.figure(figsize=(8, 6))
    plt.scatter(y_sop, preds_sop_full, color='green', alpha=0.6)
    plt.plot([y_sop.min(), y_sop.max()], [y_sop.min(), y_sop.max()], 'r--', label='Perfect Prediction')
    plt.title('SOP Estimation: Actual vs Predicted (train fit)')
    plt.xlabel('Actual SOP')
    plt.ylabel('Predicted SOP')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_sop_results.png"))
    plt.close()

    print("Result plots saved.")


if __name__ == "__main__":
    train_and_evaluate()
