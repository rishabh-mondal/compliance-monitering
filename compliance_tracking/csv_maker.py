import pandas as pd

# List of CSV file paths
csv_files = {
    'Punjab': '/home/patel_zeel/compass24/exact_latlon/punjab.csv',
    'Haryana': '/home/patel_zeel/compass24/exact_latlon/haryana.csv',
    'Bihar': '/home/patel_zeel/compass24/exact_latlon/bihar.csv',
    'Uttar Pradesh': '/home/patel_zeel/compass24/exact_latlon/uttar_pradesh.csv',
    'West Bengal': '/home/patel_zeel/compass24/exact_latlon/west_bengal.csv',
}

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through the dictionary and read the CSV files into DataFrames
for file in csv_files.values():
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
# combined_df.to_csv('combined_fil.csv', index=False)


csv1 = "/home/patel_zeel/compass24/exact_latlon/west_bengal.csv"
df1 = pd.read_csv(csv1)
df2 = df1.to_csv("file2.csv")


