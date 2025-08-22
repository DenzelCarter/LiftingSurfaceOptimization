import pandas as pd
import argparse
import os

def create_tare_lookup(raw_tare_file, output_file):
    """
    Processes a raw spinning tare run to create a clean, binned lookup table.
    """
    # Define column names (must match your raw data file)
    RPM_COL = 'Motor Electrical Speed (RPM)'
    THRUST_COL = 'Thrust (N)'
    TORQUE_COL = 'Torque (N·m)'

    print(f"Reading raw tare data from '{raw_tare_file}'...")
    try:
        df = pd.read_csv(raw_tare_file)

        # Basic cleaning
        df = df[[RPM_COL, THRUST_COL, TORQUE_COL]].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Create the RPM bins (rounded to the nearest 50 RPM)
        df['rpm_bin'] = (df[RPM_COL] / 50).round() * 50

        # Group by the bin and calculate the average thrust and torque
        print("Averaging data into RPM bins...")
        tare_lookup = df.groupby('rpm_bin')[[THRUST_COL, TORQUE_COL]].mean()

        # Rename columns to the names expected by the main script
        tare_lookup.rename(columns={
            THRUST_COL: 'thrust_tare',
            TORQUE_COL: 'torque_tare'
        }, inplace=True)

        # Save the final lookup table
        tare_lookup.to_csv(output_file)
        print(f"\nSuccess! Tare lookup table saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Raw tare file not found at '{raw_tare_file}'")
    except KeyError as e:
        print(f"Error: A required column was not found in the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a tare lookup table from a raw spinning tare run CSV."
    )
    parser.add_argument("input_file", help="Path to the raw tare run CSV file.")
    parser.add_argument(
        "--output_file",
        default="Experiment/tare_lookup.csv",
        help="Path to save the output lookup file."
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    create_tare_lookup(args.input_file, args.output_file)