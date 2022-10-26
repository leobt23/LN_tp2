from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import nltk
from sqlalchemy import column



def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return str([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


# Get text from file
text_dir = "train.txt"
text = []
f = open(text_dir, 'r')
text = f.read().splitlines()
f.close()

# Step 2 - Manipulate data
# Text to pandas.Dataframe  
df1 = pd.DataFrame(text)
df1 = df1.rename(columns={0: 'alltext'})


df1["alltext"] = df1["alltext"].str.replace(":)", "happy", regex=False)
df1["alltext"] = df1["alltext"].str.replace(":D", "happy", regex=False) # Testar sem isto depois
df1["alltext"] = df1["alltext"].str.replace(":-D", "happy", regex=False)
df1["alltext"] = df1["alltext"].str.replace(":(", "sad", regex=False)
df1["alltext"] = df1["alltext"].str.replace("):", "sad", regex=False)
df1["alltext"] = df1["alltext"].str.replace(":-(", "sad", regex=False)
df1["alltext"] = df1["alltext"].str.replace(":/", "sad", regex=False)

# Divide labels and text
df1["alltext"]= df1["alltext"].str.split("=", n = 2, expand = False)
df1[['delete', 'label','text']] = pd.DataFrame(df1.alltext.tolist(), index= df1.index)
df1 = df1[['text','label']]

# Clean "\t"
df1['text'] = df1['text'].str.replace('\t', '')

# Categories to numbers
df1['category_to_num']=df1['label'].map({'Poor':1,'Unsatisfactory':2,'Good':3,'VeryGood':4,'Excellent':5})

# Remove pontuation
df1['text'] = df1['text'].str.replace('[^a-zA-Z0-9\']', ' ', regex=True).str.strip()
#print(df1)



# Get most frequent words
list = df1['text'].values.tolist()
list_of_words = [
    word
    for phrase in list
    for word in phrase.split()
]


top_words = [x.lower() for x in list_of_words]
top_words = pd.value_counts(np.array(top_words))
print(top_words[0:30])

df1['text'] = df1.text.apply(lemmatize_text)

# Initialize the vectorizer 
stop_w_list = {'for', 'the', 'it', 'a', 'i', 'this','my', 'and', 'to', 'me', 'of', 'as'}
tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3), stop_words=stop_w_list)

doc_vec = tfidf_vectorizer.fit_transform(df1['text'])

#df2 = pd.DataFrame(doc_vec.toarray().transpose(),index=tfidf_vectorizer.get_feature_names())


# Splitting dataset into 80% training set and 20% test set
kfold = KFold(5)
for train_index, validate_index in kfold.split(df1['text'], df1['category_to_num']):
    x_train, x_test = df1['text'][train_index], df1['text'][validate_index]
    y_train, y_test = df1['category_to_num'][train_index], df1['category_to_num'][validate_index]

#converting training features into numeric vector
X_train = tfidf_vectorizer.fit_transform(x_train)
#converting training labels into numeric vector
X_test = tfidf_vectorizer.transform(x_test)

# Naive Bayes Model
mnb = MultinomialNB(alpha = 0.5)
mnb.fit(X_train,y_train)

# Results
result= mnb.predict(X_test)
print(confusion_matrix(y_test, result))
print(accuracy_score(y_test, result))


