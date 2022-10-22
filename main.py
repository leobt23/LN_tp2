from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

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
print(df1)

# Initialize the vectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
doc_vec = tfidf_vectorizer.fit_transform(df1['text'])

df2 = pd.DataFrame(doc_vec.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names())


#print(df2)
# to check if everything is working as planned
#df2.to_csv('result_df.txt', sep=' ')
#print(tfidf_vectorizer.get_feature_names())

#Splitting dataset into 60% training set and 40% test set
x_train, x_test, y_train, y_test = train_test_split(df1['text'], df1['label'], random_state=50)
#Here we convert our dataset into a Bag Of Word model using a Bigram model



#converting traning features into numeric vector
X_train = tfidf_vectorizer.fit_transform(x_train)
#converting training labels into numeric vector
X_test = tfidf_vectorizer.transform(x_test)

mnb = MultinomialNB(alpha =0.2)

mnb.fit(X_train,y_train)

result= mnb.predict(X_test)

print(accuracy_score(result,y_test))