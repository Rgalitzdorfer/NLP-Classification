import pandas as pd
from collections import defaultdict

# Paths
input_file = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Complete Data/Complete_Data_0115.csv'
output_file = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Text Distribution/Text_Distribution_0115.csv'

# Function to distribute text proportionally
def distribute_text_proportionally(df, text_column='Text'):
    text_groups = defaultdict(list)
    for idx, row in df.iterrows():
        text_groups[row[text_column]].append(idx)
    
    for text, indices in text_groups.items():
        if len(indices) > 1:
            print(f"\nOriginal Text: '{text}'")
            words = text.split()
            total_words = len(words)
            portion_size = total_words // len(indices)
            
            print(f"Total Words: {total_words}")
            print(f"Number of Rows: {len(indices)}")
            print(f"Portion Size: {portion_size}")
            
            for i, idx in enumerate(indices):
                if i == len(indices) - 1:
                    new_text = ' '.join(words[i * portion_size:])
                else:
                    new_text = ' '.join(words[i * portion_size: (i + 1) * portion_size])
                
                df.at[idx, text_column] = new_text
                print(f"Row {idx}: '{new_text}' (Words: {len(new_text.split())})")
            
            total_distributed_words = sum(len(df.at[idx, text_column].split()) for idx in indices)
            print(f"Total Distributed Words: {total_distributed_words}")
            if total_words != total_distributed_words:
                print(f"WARNING: Text loss detected! Original: {total_words}, Distributed: {total_distributed_words}")
    
    return df

# Read the specified CSV file
df = pd.read_csv(input_file)

# Print DataFrame before distribution for debugging
print("\nDataFrame before distribution:")
print(df.to_string())

# Apply the text distribution function
df = distribute_text_proportionally(df, text_column='Text')

# Print DataFrame after distribution for debugging
print("\nDataFrame after distribution:")
print(df.to_string())

# Save the modified DataFrame to a new CSV file for debugging
df.to_csv(output_file, index=False)
print(f"\nModified data saved to {output_file}")

# Reload and print the CSV to verify contents
df_loaded = pd.read_csv(output_file)
print("\nLoaded DataFrame for verification:")
print(df_loaded.to_string())

# Compare original and loaded DataFrames for consistency
print("\nComparing original and loaded DataFrames:")
print(df.equals(df_loaded))