import pandas as pd
import argparse
import os

def flip_torque_sign(file_path):
    """
    Reads a CSV file, flips the sign of the 'Torque (N·m)' column,
    and saves it to a new file.
    """
    # Define the column name to be modified
    torque_column = 'Torque (N·m)'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Reading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)

        # Check if the torque column exists in the dataframe
        if torque_column in df.columns:
            print(f"Flipping the sign of the '{torque_column}' column...")
            # Multiply the entire column by -1
            df[torque_column] = df[torque_column] * -1

            # Create the new filename
            directory, original_filename = os.path.split(file_path)
            base_name, extension = os.path.splitext(original_filename)
            new_filename = f"{base_name}_fixed{extension}"
            new_file_path = os.path.join(directory, new_filename)

            # Save the modified dataframe to the new file
            df.to_csv(new_file_path, index=False)
            print(f"\nSuccess! Modified data saved to '{new_file_path}'")
        else:
            print(f"Error: Column '{torque_column}' not found in the file.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Flip the sign of the 'Torque (N·m)' column in a CSV file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file."
    )
    
    args = parser.parse_args()
    
    # Run the function with the provided file path
    flip_torque_sign(args.input_file)