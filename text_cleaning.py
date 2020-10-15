import pandas as pd
import numpy as np
import neattext as nt
import neattext.functions as nfx

df = pd.read_csv('offensive_data.csv')
#print(df.head())

#print(df.columns)

df = df[['tweet', 'class']]
#print(df.head())

#print(df.iloc[4]['tweet'])

# Remove special characters, hastags, sopwords/punctations
# Methods/attrib
#print(dir(nt))
#print(dir(nfx))

s = df.iloc[4]['tweet']
#print(s)

# Method 1:Oop using Textframe
docx = nt.TextFrame(s)

docx.describe()
#print(docx.head(10))

# Remove stopwords
#print(docx.remove_stopwords().text)
 # Remove punctations
#print(docx.remove_puncts().text)
#print(docx.remove_puncts(most_common=False).text)


# Method2: Using Functional Approach
#print(s)

# Remove userhandles,hastags,specialcharacters
#print(nfx.remove_userhandles(s))
#print(nfx.remove_hashtags(s))
#print(nfx.remove_special_characters(s))
#print(nfx.remove_stopwords(s))

# Neat TextFrame
s2 = '!!!!!!!!!!!!! RT @ShenikaRoberts: The shit you hear about me might be true or it might be faker than the bitch who told it  to ya &#57361;'
#print(nfx.clean_text(s2))
#print(nfx.clean_text(s2,puncts=False, stopwords=False, custom_pattern=r'@\S+'))


# Noise scan
#print(df['tweet'].apply(lambda x : nt.TextFrame(x).noise_scan()['text_noise']))

# Extract userhandles
df['userhandles'] = df['tweet'].apply(nfx.extract_userhandles)
#print(df.head())

df['clean_tweet'] = df['tweet'].apply(nfx.remove_userhandles)
#print(df.head())

# Extract hashtags
#print(df['tweet'].apply(nfx.extract_hashtags))

# Remove hashtags
#print(df['tweet'].apply(nfx.remove_hashtags))

# Remove Custom pattern
df['clean_tweet']= df['clean_tweet'].apply(lambda x: nfx.remove_custom_pattern(x,term_pattern=r'&#\S+'))
#print(df['clean_tweet'])

df['clean_tweet'] = df['clean_tweet'].apply(lambda x: nfx.remove_custom_pattern(x,term_pattern=r'RT'))
#print(df['clean_tweet'])

df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_special_characters)
#print(df['clean_tweet'])

df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_multiple_spaces)
#print(df['clean_tweet'])
#print(df.head())


# Contractions
####Contractions, neattext

s = "i'm here y'all"

import contractions
contractions.fix(s)
print(nfx.fix_contractions(s))

# Extraction stopwords
df['clean_tweet'].apply(lambda x : nt.TextExtractor(x).extract_stopwords())

# Remove stopwords
# Method using TextFrame
#print(df['clean_tweet'].apply(lambda x : nt.TextExtractor(x).remove_stopwords()))

# Method2 using neattext.functions
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_stopwords)
#print(df['clean_tweet'])
print(df['clean_tweet'].apply(lambda x : nt.TextFrame(x).noise_scan()['text_noise']))
