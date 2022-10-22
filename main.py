from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import numpy as np

# Get text from file
text_dir = "train.txt"
text = []

f = open(text_dir, 'r')
text = f.read().splitlines()
f.close()

# Text to pandas.Dataframe  
df1 = pd.DataFrame(text)
df1 = df1.rename(columns={0: 'alltext'})
df1["alltext"]= df1["alltext"].str.split("=", n = 2, expand = False)
df1[['delete', 'label','text']] = pd.DataFrame(df1.alltext.tolist(), index= df1.index)
df1 = df1[['text','label']]
df1['text'] = df1['text'].str.replace('\t', '')
df1['category_to_num']=df1['label'].map({'Poor':1,'Unsatisfactory':2,'Good':3,'VeryGood':4,'Excellent':5})
df1['text'] = df1['text'].str.replace('[^a-zA-Z0-9\']', ' ', regex=True).str.strip()
print(df1)

list = df1['text'].values.tolist()
#print(list)
#print(pd.value_counts(np.array(list)))
list_of_words = [
    word
    for phrase in list
    for word in phrase.split()
]
#print(list_of_words)
top_words = [x.lower() for x in list_of_words]

top_words = pd.value_counts(np.array(top_words))

print(top_words[0:20])

# Initialize the vectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3))
doc_vec = tfidf_vectorizer.fit_transform(df1['text'])

df2 = pd.DataFrame(doc_vec.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names())


#print(df2)
# to check if everything is working as planned
#df2.to_csv('result_df.txt', sep=' ')
#print(tfidf_vectorizer.get_feature_names())

#Splitting dataset into 60% training set and 40% test set
kfold = KFold(5)
for train_index, validate_index in kfold.split(df1['text'], df1['category_to_num']):
    x_train, x_test = df1['text'][train_index], df1['text'][validate_index]
    y_train, y_test = df1['category_to_num'][train_index], df1['category_to_num'][validate_index]
#Here we convert our dataset into a Bag Of Word model using a Bigram model



#converting traning features into numeric vector
X_train = tfidf_vectorizer.fit_transform(x_train)
#converting training labels into numeric vector
X_test = tfidf_vectorizer.transform(x_test)

mnb = MultinomialNB(alpha = 0.2)

mnb.fit(X_train,y_train)

result= mnb.predict(X_test)
#print(result)
#print(confusion_matrix(result, y_test))

#print(accuracy_score(result,y_test))