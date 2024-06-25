#Import Libraries
import pandas as pd #DataFrames
import os #Operating System
import nltk #Text Processing
from collections import Counter #Tracking Occurences
from nltk.corpus import stopwords #Accessing Stopwords
from nltk.tokenize import word_tokenize #Tokenize String into Words

#Directories
nltk.data.path.append('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/NLTK')  
input_file = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/All_Participants.csv'  
output_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5'  
output_file = os.path.join(output_folder, 'Feature_Matrix.csv')  

#Set Up Preprocessing
df = pd.read_csv(input_file) #Load DataFrame
print(df.head()) #Print Top Rows
if 'Text' not in df.columns: #Error Detection
    raise ValueError("The 'Text' column is missing from the DataFrame.")  
df['Text'] = df['Text'].astype(str) #Convert to String
stop_words = set(stopwords.words('english')) #English Stopwords

#Preprocessing
def preprocess(text):
    text = text.lower() #Convert Text to Lowercase
    tokens = word_tokenize(text) #Tokenize Text Into Words
    tokens = [word for word in tokens if word.isalpha()] #Keep Only Alphabetic Tokens
    tokens = [word for word in tokens if word not in stop_words] #Remove Stopwords
    return tokens  

df['Tokens'] = df['Text'].apply(preprocess) #Apply Preprocessing
all_tokens = [token for tokens in df['Tokens'] for token in tokens] #Go Through All Tokens
token_counts = Counter(all_tokens) #Count Frequency of Each Token
most_common = 1000  #Number of Most Common Tokens to Keep
common_tokens = [token for token, _ in token_counts.most_common(most_common)] #Get Most Common Tokens

#Create Feature Matrix (Bag of Words)
def create_bow(tokens): 
    bow = Counter(tokens) #Count Tokens in Current Row
    return [bow.get(token, 0) for token in common_tokens] 
df_bow = df['Tokens'].apply(create_bow) #Apply BOW Function
bow_df = pd.DataFrame(df_bow.tolist(), columns=common_tokens) #Put Vectors in New DataFrame
combined_df = pd.concat([df.drop(columns=['Text', 'Tokens']), bow_df], axis=1) #Merge With Original DataFrame to Keep Relevant Data
combined_df.to_csv(output_file, index=False) #Save to CSV 

#Print Statements
print(f"Feature Matrix Saved to {output_file}") 
print(combined_df.head()) 
