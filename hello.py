import pandas as pd

# Import World Bank data from CSV
# Replace 'Average_Schooling_world.csv' with the name of your data file
data_path = 'Data /Raw Data /Average_Schooling_world.csv'
df = pd.read_csv(data_path)

print("Data imported successfully!")
print(f"Shape: {df.shape}")
print("First 5 rows:")
print(df.head())