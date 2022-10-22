from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Get text from file
text_dir = "train.txt"
text = []

f = open(text_dir, 'r')
text = f.read().splitlines()
f.close()

# Text to pandas.Dataframe  
df1 = pd.DataFrame(text)
df1 = df1.rename(columns={0: 'alltext'})
print(df1)

# Initialize the vectorizer
tfidf_vectorizer = TfidfVectorizer()
doc_vec = tfidf_vectorizer.fit_transform(df1['alltext'])

df2 = pd.DataFrame(doc_vec.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names())


print(df2)
# to check if everything is working as planned
#df2.to_csv('result_df.txt', sep=' ')
#print(tfidf_vectorizer.get_feature_names())