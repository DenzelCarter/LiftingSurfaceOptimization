import pandas as pd
import os
import re

# ===================================================================
# --- 1. SETUP: Define folder and file paths ---
# ===================================================================
INPUT_FOLDER = 'Experiment/airfoil_data'
OUTPUT_FILE = 'Experiment/tools/naca0012_all_re.csv'

# ===================================================================
# --- 2. Main Script to Read, Process, and Combine Files ---
# ===================================================================
all_dataframes = []

print(f"Reading all data files from '{INPUT_FOLDER}'...")

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(('.csv', '.txt')):
        match = re.search(r'(\d+)', filename)
        if match:
            reynolds_number = int(match.group(1))
            file_path = os.path.join(INPUT_FOLDER, filename)
            
            try:
                # --- FINAL ROBUST PARSING METHOD ---
                # The user confirmed a comma separator and a single header line.
                # Using the 'python' engine is more robust to formatting issues.
                df = pd.read_csv(
                    file_path, 
                    sep=',',           # Use comma as the separator
                    skiprows=1,        # Skip the single header row
                    engine='python'    # Use the more flexible python engine
                )

                # Ensure the column names are correct
                df.columns = ['alpha', 'cl', 'cd', 'cdp', 'cm', 'top_xtr', 'bot_xtr']
                
                df['reynolds'] = reynolds_number
                all_dataframes.append(df)
                print(f"  - Processed {filename} for Re = {reynolds_number}")

            except Exception as e:
                print(f"  - Error processing {filename}: {e}. Skipping.")

        else:
            print(f"  - Warning: Could not find Reynolds number in filename '{filename}'. Skipping.")

if all_dataframes:
    master_df = pd.concat(all_dataframes, ignore_index=True)
    final_df = master_df[['reynolds', 'alpha', 'cl', 'cd']]
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully combined {len(all_dataframes)} files into '{OUTPUT_FILE}'")
else:
    print("\nNo data files were processed.")