import pandas as pd

# Load both CSV files
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")

# Combine them (stack rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new CSV
combined_df.to_csv("combined.csv", index=False)
