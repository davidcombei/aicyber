import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('emails.csv')
df.drop_duplicates(inplace = True)
df.isnull().sum()
nltk.download('stopwords')


def process_text(text):

   #1. Stergerea caracterelor de punctuatie
   #2. Stergerea cuvintelor cheie
   #3. Returnare lista de jetoane

   #1.
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

   #2.
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  #3.
    return clean_words

## tokenizer
count_vec=CountVectorizer(analyzer=process_text)
messages = count_vec.fit_transform(df['text'])


## train-test split with 80-20
X_train, X_test, y_train, y_test = train_test_split(messages, df['spam'],test_size=0.20, random_state=0)
print("train size:",X_train.shape)
print("test size:",X_test.shape)

## SVM classifier
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, y_train)

print('#### metrics:\n')
pred_svm = svm.predict(X_test)

print(classification_report(y_test, pred_svm))
print()
print('Confusion Matrix: \n',confusion_matrix(y_test, pred_svm))
print()
print('Accuracy: ', accuracy_score(y_test, pred_svm))
joblib.dump(svm, "svc_spam.joblib")
joblib.dump(count_vec, "vectorizer.joblib")
