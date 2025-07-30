import pandas as pd
import argparse
import sys

def create_comparison_csv(datasets_to_compare, output_filename):
    """
    Compares specified datasets and creates a CSV of triggers with valid
    measurements in at least two of the chosen sets.
    """
    file_map = {
        'all': 'Trigger_number_vs_mvt_all_det.csv',
        'best': 'Trigger_number_vs_mvt_best_det_2.csv',
        'one': 'Trigger_number_vs_mvt_one_det.csv'
    }
    
    df_list = []
    print("Reading data files...")
    for key, filename in file_map.items():
        try:
            df = pd.read_csv(filename).set_index('trigger_number')
            # --- THE FIX ---
            # Add a suffix to ALL columns to prevent name clashes during concat
            df = df.add_suffix(f'_{key}')
            df_list.append(df)
        except FileNotFoundError:
            print(f"❌ Error: The file '{filename}' was not found.", file=sys.stderr)
            sys.exit(1)
            
    # Combine all data into a master DataFrame. Columns are now unique (e.g., T90_all, T90_best).
    combined_df = pd.concat(df_list, axis=1, join='outer')

    # Consolidate the property columns now that they have unique names
    property_cols = ['T90', 'T50', 'PF64', 'PFLX', 'FLUxe6']
    for col in property_cols:
        # This will now correctly find ['T90_all', 'T90_best', 'T90_one']
        source_cols = [f'{col}_{key}' for key in file_map.keys() if f'{col}_{key}' in combined_df.columns]
        if source_cols:
            # Get the first non-null value from the source columns for each row
            combined_df[col] = combined_df[source_cols].bfill(axis=1).iloc[:, 0]
            # Drop the original suffixed columns
            combined_df.drop(columns=source_cols, inplace=True)
    
    print(f"Filtering for triggers with measurements in at least two of: {', '.join(datasets_to_compare)}")
    conditions = []
    for key in datasets_to_compare:
        error_col = f'mvt_error_ms_{key}'
        # Check if the column exists before trying to access it
        if error_col in combined_df.columns:
            conditions.append((combined_df[error_col] > 0).fillna(False))

    if not conditions:
        print("⚠️ No valid data columns to compare.")
        return

    num_valid_measurements = pd.concat(conditions, axis=1).sum(axis=1)
    filtered_df = combined_df[num_valid_measurements >= 2].copy()
    
    if filtered_df.empty:
        print("⚠️ No triggers found that match the criteria.")
        return

    # Arrange columns for the final output
    ms_cols = [f'mvt_ms_{key}' for key in datasets_to_compare]
    error_cols = [f'mvt_error_ms_{key}' for key in datasets_to_compare]
    final_cols = property_cols + ms_cols + error_cols
    
    # Select only the columns that actually exist in the filtered DataFrame
    existing_final_cols = [c for c in final_cols if c in filtered_df.columns]
    
    final_df = filtered_df[existing_final_cols].reset_index()
    
    final_df.to_csv(output_filename, index=False, float_format='%.6g')
    
    print(f"\n✅ Success! {len(final_df)} triggers found.")
    print(f"Comparison results saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MVT measurements from different datasets.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--compare', nargs='+', choices=['all', 'best', 'one'], required=True,
                        help='A list of 2 or 3 datasets to compare (e.g., -c all best).')
    
    parser.add_argument('-o', '--output', default='comparison.csv',
                        help='Name for the output CSV file.')

    args = parser.parse_args()

    if len(args.compare) < 2:
        print("❌ Error: Please specify at least two datasets to compare.", file=sys.stderr)
        sys.exit(1)
        
    create_comparison_csv(args.compare, args.output)