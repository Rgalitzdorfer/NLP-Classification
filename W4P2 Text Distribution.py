#Import Libraries
import os #Operating Systems
import pandas as pd #DataFrames
from collections import defaultdict #Dictionary
 
#Directories
input_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Complete Data' #Get data from this folder
output_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Text Distribution' #Save data to this folder
os.makedirs(output_folder, exist_ok=True) #Ensure directory exists

#Distribute Text Proportionally using Cumulative Time
def distribute_text_proportionally(df, text_column='Text', time_column='Cumulative_Updated_True_Time'):
    text_groups = defaultdict(list) #Group rows if they have identical text
    for idx, row in df.iterrows():
        text_groups[row[text_column]].append(idx)
    for text, indices in text_groups.items():
        if len(indices) > 1: #Process if there is more than one row with identical text
            words = text.split() #Split words if there is a space between them
            total_words = len(words) #Keep track to prevent data slippage
            total_time = df.loc[indices, time_column].sum() #Sum total time for distribution
            proportions = df.loc[indices, time_column] / total_time #Get proportions
            current_index = 0 #Initialize
            for i, (idx, proportion) in enumerate(zip(indices, proportions)): #Assign proportionate number of words to each row, 
                if i == len(indices) - 1: #Assign remaining words to the last row
                    new_text = ' '.join(words[current_index:])
                else:
                    num_words = round(proportion * total_words) #Round if necessary
                    new_text = ' '.join(words[current_index:current_index + num_words])
                    current_index += num_words #Accumulator
                
                df.at[idx, text_column] = new_text
            total_distributed_words = sum(len(df.at[idx, text_column].split()) for idx in indices) #Used for Comparison
            if total_words != total_distributed_words: #Error Detection
                print(f"WARNING: Text loss detected! Original: {total_words}, Distributed: {total_distributed_words}")
            else:
                print(f"SUCCESS: Text distributed correctly! Original: {total_words}, Distributed: {total_distributed_words}")
    return df

#Read Each CSV in the Input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, f'Text_Distribution_{filename}')
        try:
            df = pd.read_csv(input_file_path)
            text_column = 'Text'  #Ensure text Column exists
            df = distribute_text_proportionally(df, text_column=text_column) #Apply Text Distribution
            df.to_csv(output_file_path, index=False) #Save to new folder
            print(f"\nModified Data Saved to {output_file_path}")
            df_loaded = pd.read_csv(output_file_path)
            original_word_count = sum(df_loaded[text_column].apply(lambda x: len(str(x).split()))) #Old word count
            new_word_count = sum(df[text_column].apply(lambda x: len(str(x).split()))) #New word count
            print(f"Original Word Count: {original_word_count}, New Word Count: {new_word_count}") #Error Detection
            if original_word_count == new_word_count:
                print(f"SUCCESS: Word Counts Match for {filename}!") #Ensure word counts are identical
            else:
                print(f"ERROR: Word Counts Do Not Match for {filename}!") #Error Detection
        except Exception as e:
            print(f"Error processing {filename}: {e}")
print("All Files Processed.")
