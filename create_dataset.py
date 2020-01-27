# import libraries
import pandas as pd


# reading data file
df = pd.read_csv("/Users/hlab/Documents/GitHub/primal/dataset/primals_clean3_filtered.csv")
print(df['first_sentence_primal'].head())
df['perfect_text'] = 'the world is ' + df['first_sentence_primal'].str.lower()


# reduce duplicates for long and highly repeatitive texts
perfect_text = df["perfect_text"]
duplicated_text = perfect_text[perfect_text.isin(perfect_text[perfect_text.duplicated()])]
duplicated_text = duplicated_text.drop_duplicates()
unduplicated_text = perfect_text[~(perfect_text.isin(perfect_text[perfect_text.duplicated()]))]
reduced_duplicated_text = []
COUNT_threshold = 15    
LENGTH_threshold = 10   
KEEP_rate = 0.2        
for text in list(duplicated_text):
    text_length = len(text.split()) 
    count_text = list(perfect_text).count(text)
    
    # only reduce duplicates when passing COUNT_threshold and LENGTH_threshold
    if (text_length>=LENGTH_threshold) and (count_text>= COUNT_threshold):
        print("deteceted text: {} - count: {}".format(text, str(count_text)))
        count_kept = round(count_text*KEEP_rate)
    else:    
        count_kept = count_text
    
    # add to reduced_duplicated_text
    reduced_duplicated_text.extend([text] * count_kept)
# add unduplicated_text to reduced_duplicated_text        
reduced_duplicated_text = reduced_duplicated_text + list(unduplicated_text)   


# shuffle
import random 
random.shuffle(reduced_duplicated_text)
print("len reduced_duplicated_text: " + str(len(reduced_duplicated_text)))


# filter and write to text file
output_file = "/Users/hlab/Documents/GitHub/primal/dataset/perfect_clean3.txt"
writer = open(output_file, "w")
for text in reduced_duplicated_text:
 
      # filtering irrelevant primals
      if "jonas" in text:
          continue
      if "changing nd" in text:
          text = text.replace("changing nd", "changing and")
      if "<newline>" in text:
          text = text.replace("<newline>", " ")
      if "(feat" in text:
          text = text.replace("(feat", " ")    
      if "&amp" in text:
          text = text.replace("&amp;", "and")            
          
      # write to file  
      writer.write(text + "\n")
writer.close()

