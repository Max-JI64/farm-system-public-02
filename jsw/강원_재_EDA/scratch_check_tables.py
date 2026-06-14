import pandas as pd
import glob
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# List files in outputs/Step3/tables
table_dir = "D:/farm-system-public-02/jsw/강원_재_EDA/outputs/Step3/tables"
files = glob.glob(os.path.join(table_dir, "*.csv"))
print("Step3 table files:")
for f in files:
    name = os.path.basename(f)
    print(f"- {name}")
    try:
        df = pd.read_csv(f, nrows=1, encoding='utf-8-sig')
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"  Error reading: {e}")
