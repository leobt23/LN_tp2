from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import nltk
import matplotlib.pyplot as plt
import sys

# Get text from file
text = sys.argv[2]
f = open(text, 'r', encoding='utf-8', errors='ignore')
text_1 = f.read().splitlines()
f.close()

text = sys.argv[4]
f = open(text, 'r', encoding='utf-8', errors='ignore')
text_2 = f.read().splitlines()
f.close()

if(sys.argv[1] == "-test"):
    test_text = text_1
elif(sys.argv[1] == "-train"):
    train_text = text_1

if(sys.argv[3] == "-test"):
    test_text = text_2
elif(sys.argv[3] == "-train"):
    train_text = text_2

def preprocessing(text, flag_train):

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
    if(flag_train == True):
        df1["alltext"]= df1["alltext"].str.split("=", n = 2, expand = False)
        df1[['delete', 'label','text']] = pd.DataFrame(df1.alltext.tolist(), index= df1.index)
        df1 = df1[['text','label']]
    else:
        df1[['text']] = pd.DataFrame(df1.alltext.tolist(), index= df1.index)

    # Clean "\t"
    df1['text'] = df1['text'].str.replace('\t', '')

    if(flag_train == True):
        # Categories to numbers
        df1['category_to_num']=df1['label'].map({'Poor':1,'Unsatisfactory':2,'Good':3,'VeryGood':4,'Excellent':5})

    # Remove pontuation
    df1['text'] = df1['text'].str.replace('[^a-zA-Z0-9\']', ' ', regex=True).str.strip()
    return df1

# Initialize the vectorizer 
stop_w_list = {'for', 'the', 'it', 'a', 'i', 'this','my', 'and', 'to', 'me', 'of', 'as'}
tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3), stop_words=stop_w_list)


# Splitting dataset into 80% training set and 20% test set
"""
kfold = KFold(5)
accuracy_kfold = []
accuracy_kfold2 = []
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
    accuracy_kfold.append(accuracy_score(y_test, result))
    
    #Confusion matrix (CHECK IF WE CAN USE MATPLOTLIB)
    cm = confusion_matrix(y_test, result)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


print(accuracy_kfold)
print(np.mean(accuracy_kfold))
"""

#Final training

df_train = preprocessing(text = train_text, flag_train=True)
df_test = preprocessing(text = test_text, flag_train=False)
mnb = MultinomialNB(alpha = 0.5)
X = tfidf_vectorizer.fit_transform(df_train['text'])
y = df_train['category_to_num']
mnb.fit(X,y)

X_test = tfidf_vectorizer.fit_transform(df_train['text'])
result= mnb.predict(X_test)

result2 = []

for i in result:
    if(i == 1):
        result2.append("=Poor=")
    if(i == 2):
        result2.append("=Unsatisfactory=")
    if(i == 3):
        result2.append("=Good=")
    if(i == 4):
        result2.append("=VeryGood=")
    if(i == 5):
        result2.append("=Excellent=")


for line in result2:
    print(line)

