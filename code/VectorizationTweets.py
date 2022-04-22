import main
import pandas as pd

def BagOfWords():
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=1000,
    stop_words='english')
    vector = vectorizer.fit_transform(Data['text'])
    df_bow_sklearn = pd.DataFrame(vector.toarray(),columns=vectorizer.get_feature_names_out())
    SaveFile(df_bow_sklearn,'bagofwords.pkl')

def TF_IDF():
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.feature_extraction.text import TfidfVectorizer
   tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000,
    stop_words='english')
   tfidf_vector = tfidf_vectorizer.fit_transform(Data['text'])
   tfidf_df = pd.DataFrame(tfidf_vector.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
   SaveFile(tfidf_df,'TF-IDF.pkl')
