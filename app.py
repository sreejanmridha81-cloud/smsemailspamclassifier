import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
ps=PorterStemmer()
from nltk.corpus import stopwords
import string


STOPS = set(stopwords.words("english"))



def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum() and (i not in STOPS) and (i not in string.punctuation):
            y.append(ps.stem(i))

    return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/Sms classifier")
input_sms=st.text_area("enter your message")
if st.button("predict"):
    # 1. preprocess (indented)
    transformed_sms = transform_text(input_sms)

    # 2. vectorize (indented)
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict (indented)
    result = model.predict(vector_input)[0]

    # 4. display (indented)
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")











