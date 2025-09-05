import pandas as pd
import numpy as np
import argparse
import os

def analyze_sample_rate(file_path):
    """
    Analyzes the sampling rate from the first column of a CSV file.

    Args:
        file_path (str): The full path to the input CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Analyzing timestamps from '{file_path}'...")

    try:
        # Load the CSV file
        df = pd.read_csv(file_path, low_memory=False)

        if df.empty:
            print("Error: The CSV file is empty.")
            return

        # --- Data Cleaning ---
        # Get the first column
        time_col_name = df.columns[0]
        
        # Convert to numeric, coercing any non-numeric values (like headers) to NaN
        timestamps_series = pd.to_numeric(df[time_col_name], errors='coerce')
        
        # Drop any NaN values and remove duplicates
        timestamps_series.dropna(inplace=True)
        timestamps_series.drop_duplicates(inplace=True)
        
        if len(timestamps_series) < 2:
            print("Error: Not enough valid timestamp data points to analyze.")
            return

        timestamps = timestamps_series.to_numpy()

        # --- Calculations ---
        # Calculate the time difference between each sample
        deltas = np.diff(timestamps)
        
        # Filter out any non-positive deltas to avoid errors
        deltas = deltas[deltas > 0]

        if len(deltas) == 0:
            print("Error: No valid time intervals found between samples.")
            return
            
        # Calculate the instantaneous frequencies
        frequencies = 1 / deltas

        # --- Report Metrics ---
        print("\n--- Sampling Rate Analysis ---")
        print(f"Total Valid Samples: {len(timestamps):,}")
        print(f"Total Duration:      {timestamps[-1] - timestamps[0]:.2f} seconds")
        print("-" * 30)
        print(f"Average Frequency:   {np.mean(frequencies):.2f} Hz")
        print(f"Standard Deviation:  {np.std(frequencies):.2f} Hz")
        print(f"Minimum Frequency:   {np.min(frequencies):.2f} Hz")
        print(f"Maximum Frequency:   {np.max(frequencies):.2f} Hz")
        print("------------------------------\n")
        
        if np.std(frequencies) > (np.mean(frequencies) * 0.05): # If std is >5% of mean
             print("Conclusion: The sampling rate is NOT stable.")
        else:
             print("Conclusion: The sampling rate is stable.")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze the sampling rate from the first column (timestamps) of a CSV file."
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the CSV file to be analyzed."
    )
    
    args = parser.parse_args()
    
    # Run the analysis function with the provided file path
    analyze_sample_rate(args.input_file)