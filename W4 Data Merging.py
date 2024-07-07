#Import Libraries
import os #Operating Systems
import pandas as pd #DataFrames
from collections import defaultdict #Dictionary

#Directories
input_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data' #Input
output_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Complete Data' #Output
os.makedirs(output_folder, exist_ok=True) #Ensure they exist

#Process each file
def process_file(file_path):
    df = pd.read_csv(file_path)
    print(f"Processing {file_path}") #Error Detection
    print(f"Columns: {df.columns.tolist()}") #Error Detection
    problem_col = 'ProblemID' 
    step_col = 'ProblemStepID'
    df['Text'] = df['Text'].astype(str).apply(lambda x: ' '.join(x.split())) #Convert to string & remove extra spaces
    if 'State' in df.columns:
        df['State'] = df['State'].fillna(' ') #Fill Nan values
    if 'Rule_order' in df.columns:
        df['Rule_order'] = df['Rule_order'].fillna(' ') #Fill Nan values
    timestamp_columns = ['Cumulative_Updated_True_Time', 'Timestamp_seconds'] #Column Identification
    text_columns = ['Text', 'State'] #Column Identification
    updated_true_time_column = 'Updated_True_Time' #Column Identification
    columns_to_remove_duplicates = [col for col in df.columns if col not in text_columns + timestamp_columns + [updated_true_time_column, problem_col, step_col]] #Remove Duplicates for Some Columns
    grouped = df.groupby([problem_col, step_col]) #Group by Problem & Problem Step
    #Concatenate Text
    def concatenate_text_state(series):
        return ' '.join(series.dropna().astype(str))
    #Remove Duplicates, Concatenate Values
    def remove_duplicates(series):
        return ', '.join(sorted(set(series.dropna().astype(str))))
    #Trim at '||' Delimiter
    def concatenate_and_remove_duplicates(text_series):
        texts = text_series.dropna().unique()
        concatenated_text = ' || '.join(texts)
        return concatenated_text.split(' || ')[0] #Trim at the first occurence
    #Aggregate the data
    aggregation_dict = {
        'Text': concatenate_and_remove_duplicates,
        'State': concatenate_text_state if 'State' in df.columns else 'first',
        'Cumulative_Updated_True_Time': 'last',
        'Timestamp_seconds': 'last',
        'Rule_order': lambda x: ', '.join(sorted(set(x.dropna().astype(str)))),
        'Updated_True_Time': lambda x: round(x.dropna().astype(float).sum(), 2), #Round to 2 Decimal Places
        'Rule_type': lambda x: ', '.join(sorted(set(x.dropna().astype(str)))),
        'Participant': 'first'
    }
    #Add columns to aggregation dictionary
    for col in columns_to_remove_duplicates:
        if col in df.columns:
            aggregation_dict[col] = remove_duplicates
    aggregated_df = grouped.agg(aggregation_dict).reset_index() #Reset Index to avoid incorrect order
    return aggregated_df


#Process Each CSV File in Input Folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        try:
            transformed_df = process_file(file_path)
            participant_id = transformed_df['Participant'].iloc[0]
            participant_number = ''.join(filter(str.isdigit, participant_id)) 
            output_filename = f"Complete_Data_{participant_number}.csv"
            output_file_path = os.path.join(output_folder, output_filename)
            transformed_df.to_csv(output_file_path, index=False)
            print(f'\nTransformed data saved to {output_file_path}')
        except KeyError as e: #Error Detection
            print(f"Skipping {filename} due to missing column: {e}")
        except ValueError as e: #Error Detection
            print(f"Skipping {filename} due to ValueError: {e}")
        except Exception as e: #Error Detection
            print(f"Skipping {filename} due to unexpected error: {e}")
print("Data Transformation Complete.")
 
#Directories For Text Distribution
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
