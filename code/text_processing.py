import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re, string

nltk.download('punkt')
nltk.download('wordnet')

def preprocess(sent):
    sent = sent.lower() 
    sent=sent.strip()  
    sent= re.compile('<.*?>').sub('', sent) 
    sent = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', sent)  
    sent = re.sub('\s+', ' ', sent)  
    sent = re.sub(r'\[[0-9]*\]',' ',sent) 
    sent=re.sub(r'[^\w\s]', '', str(sent).lower().strip())
    sent = re.sub(r'\d',' ',sent) 
    sent = re.sub(r'\s+',' ',sent) 
    return sent

def stopword(sent):
    sent = [i for i in sent.split() if i not in stopwords.words('english')]
    return ' '.join(sent)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatizer(sent, wl):
    word_pos_tags = nltk.pos_tag(word_tokenize(sent)) # Get position tags
    sent = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(sent)

df = pd.read_csv("../data/test.csv")
df = df[df['discourse_type'] == 'Claim']
df.to_csv("../data/testOnlyClaim.csv", index=False)
input()


wl = WordNetLemmatizer()
for id in range(len(df)):
    sent = df["discourse_text"][id]
    sent = preprocess(sent)
    sent = stopword(sent)
    sent = lemmatizer(sent, wl)
    df["discourse_text"][id] = sent
    id += 1
    stars = '*'*int(50*id/len(df))
    print("Processing data points: |{:50s}| {:.2f}% [{}|{}]".format(
        stars, 
        100*id/len(df), 
        id, 
        len(df))
        , end="\r")
    
print()
print("Finished")
df = df.dropna()
df.to_csv("../data/new_test.csv", index=False)
