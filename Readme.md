Data Cleaning & Preprocessing Pipeline
The raw data originated as multiple text files (10 files per OEM) distributed across zipped folders. The following pipeline was executed to create the unified CSV files (1.csv, 2.csv, 3.csv) used in this analysis.

Step 1: Data Merging

Input: 10 raw text (.txt) files per OEM folder, containing data for 256 cells.

Process: * Iterated through all files within each OEM directory.

Concatenated them into a single Dataframe per OEM.

Ensured no data loss during concatenation.

Step 2: Fixing "Tab Shift" & Column Alignment

The raw data contained structural inconsistencies that required correction:

Dummy Header Removal: The first row of the raw files contained a garbage sequence (0, 1, 2, ... 11). This was programmatically removed (skiprows=1) to expose the actual data.

Col 7 → Voltage_mV

Col 8 → Current_mA

Col 9 → Capacity_mAh

Col 10 → Energy_mWh (Identified as "Feature 10")

Col 11 → Power_W (Identified via correlation with V×I)

tep 4: State Identification

Logic: Since the "Step Number" column was not always consistent, we derived the battery state directly from the Current flow:

Charge: Current > 50 mA

Discharge: Current < -50 mA

Rest: -50 mA <= Current <= 50 mA

Goal: Estimate SOH and SOP for OEM 1 using charging data.

Result: Achieved < 2% MAPE using a Random Forest Regressor.
