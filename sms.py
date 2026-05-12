import pandas as pd
import numpy as np
df = pd.read_csv("spam.csv", encoding='latin-1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, errors='ignore')
df.rename(columns={"v1":'target',"v2":"text"},inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["target"]=le.fit_transform(df["target"])
df.isnull().sum()
df.duplicated().sum()
df=df.drop_duplicates(keep="first")
import matplotlib.pyplot as plt
# plt.pie(df["target"].value_counts(0),labels=["ham","spam"],autopct="%0.2f")
# plt.show()
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df["num_char"]=df["text"].apply(len)

df["num_word"]=(df["text"].apply(lambda x: len(nltk.word_tokenize(x))))
df["sentence_num"]=df["text"].apply(lambda x: len(nltk.sent_tokenize(x)))
# print(df[["num_char","num_word","sentence_num"]].describe())
# print(df[df["target"]==0][["num_char","num_word","sentence_num"]].describe())
# print(df[df["target"]==1][["num_char","num_word","sentence_num"]].describe())
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords
import string
string.punctuation


plt.figure(figsize=(12,6))
# sns.histplot(df[df['target'] == 0]['num_char'])
# sns.histplot(df[df['target'] == 1]['num_char'],color='red')

# plt.show()
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

STOPS = set(stopwords.words("english"))

def transform_text(text):
    
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    
    y = []
    for i in tokens:
        if i.isalnum() and (i not in STOPS) and (i not in string.punctuation):
            y.append(ps.stem(i))
            
    return " ".join(y)      
 
df["transformed_text"] = df["text"].apply(transform_text)
from wordcloud import WordCloud
wc=WordCloud(height=500,width=500,background_color="white",min_font_size=10)
spam_wc=wc.generate(df[df["target"]==1]["transformed_text"].str.cat(sep=" "))
# plt.imshow(spam_wc)
ham_wc=wc.generate(df[df["target"]==0]["transformed_text"].str.cat(sep=" "))
# plt.imshow(ham_wc)
# plt.show()
spam_corpus=[]
for msg in df[df["target"]==1]["transformed_text"].tolist():
    for word in msg.split():
        spam_corpus.append(word)

# print(len(spam_corpus))    
from collections import Counter
data=pd.DataFrame(Counter(spam_corpus).most_common(30) )  
sns.barplot(x=data[0],y=data[1])
plt.xticks(rotation="vertical")
# plt.show()

#model building
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tf=TfidfVectorizer(max_features=3000)
X=tf.fit_transform(df["transformed_text"]).toarray()
y=df["target"].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
gb=GaussianNB()
mb=MultinomialNB()
br=BernoulliNB()
gb.fit(X_train,y_train)
y_pred1=gb.predict(X_test)
print("accuracy score1= ",accuracy_score(y_test,y_pred1))
print("confusion matrix=",confusion_matrix(y_test,y_pred1))
print("precise_score",precision_score(y_test,y_pred1))

mb.fit(X_train,y_train)
y_pred2=mb.predict(X_test)
print("accuracy score 2=",accuracy_score(y_test,y_pred2))
print("confusion matrix=",confusion_matrix(y_test,y_pred2))
print("precise_score",precision_score(y_test,y_pred2))
br.fit(X_train,y_train)
y_pred3=br.predict(X_test)
print("accuracy score 3=",accuracy_score(y_test,y_pred3))
print("confusion matrix=",confusion_matrix(y_test,y_pred3))
print("precise_score",precision_score(y_test,y_pred3))
#tfid>>mb
import pickle
pickle.dump(tf,open("vectorizer.pkl","wb"))
pickle.dump(mb,open("model.pkl","wb"))














