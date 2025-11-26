Objective 2: Comparative Analysis of OEM Data

Goal
Evaluate the quality and performance characteristics of cells from three different manufacturers (OEMs) to determine their suitability for second-life applications.

Methodology
1.  Metric 1: Capacity (mAh): Defined as the maximum charge delivered by the cell during the discharge cycle.
    Indicates Energy Density and total runtime.
2.  Metric 2: Internal Resistance (DCIR): Calculated using the voltage sag method ($\Delta V / \Delta I$) at the moment the load is applied (Rest-to-Discharge transition).
    Indicates Power Capability and **Health** (SOH).

How to Run
1. Ensure all CSV files (`1.csv`, `2.csv`, `3.csv`) are in `../data/`.
2. Run the script:
   ```bash
   python comparative_analysis.py
