# Build Model Script

import os
import pickle
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pymongo import MongoClient

VERSION = "1_"

cwd = os.getcwd()


def word_tokenize_lemma_verb(text):
    words = TextBlob(text).words
    return [word.lemmatize(pos = "v") for word in words]


vect = CountVectorizer(
				  lowercase=True,
                  stop_words="english", 
                  analyzer=word_tokenize_lemma_verb,
                  ngram_range = (1,2),
                  max_features=10000
            )


def get_feature_importance(PAYLOAD_TEXT, VECT, X_TRAIN_DTM, Y_TRAIN):

    featureImportance = pd.DataFrame(data = np.transpose((clf.fit(X_TRAIN_DTM, Y_TRAIN).coef_).astype("float32")), columns = ['featureImportance'], 
             index=VECT.get_feature_names()).reset_index()
    featureImportance.columns = ['tokens', 'featureImportance']
    payload_features = pd.DataFrame(VECT.transform([PAYLOAD_TEXT]).toarray(), columns=VECT.get_feature_names(), index=['values']).T
    payload_features = payload_features[payload_features.values>0].reset_index()    
    payload_features.columns = ['tokens', 'values']
    good_features = featureImportance.merge(payload_features['tokens'], on="tokens", how='inner').sort_values('featureImportance', ascending=False)
    st.write("Here are the features of importance:")
    return good_features


# Connect to MongoDB
client = MongoClient()
# client.list_database_names()
db = client.yelpdb
collection = db.prod


# GLOBALS
n_good_bad_samples = 50000
all_stars = [1.0,2.0,3.0,4.0,5.0]
# good and bad star ratings for creating target variable
BAD = [1.0,2.0]
GOOD = [4.0,5.0]



def load_data(NUM_REVIEWS_EACH, STARS):
    df_rw_good = pd.DataFrame(list(collection.find({"stars":{"$in":[k for k in STARS if k in GOOD]}}, {"text": 1, 'stars':1}).limit(NUM_REVIEWS_EACH)))
    # st.write([k for k in STARS if k in BAD])
    df_rw_bad = pd.DataFrame(list(collection.find({"stars":{"$in":[k for k in STARS if k in BAD]}}, {"text": 1, 'stars':1}).limit(NUM_REVIEWS_EACH)))
    df_rw = pd.concat([df_rw_bad, df_rw_good], axis=0)
    df_rw.reset_index(inplace=True, drop=True)
    del df_rw_bad
    del df_rw_good
    df_rw = df_rw.astype(str)

    def good_bad_review(x):
        """
        Reviews with star ratings 4 and above are considered "Good Reviews",
        and all other reviews are considered "Bad Reviews". This function will
        create a target variable in this binary classification approach
        """
        if x >= 4:
            return 1
        else:
            return 0

    df_rw['stars'] = df_rw['stars'].astype(float)
    df_rw['target'] = df_rw['stars'].apply(good_bad_review)
    # define X and y
    X = df_rw.text
    y = df_rw.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


    return df_rw, X_train, X_test, y_train, y_test, X, y


df_rw, X_train, X_test, y_train, y_test, X, y = load_data(n_good_bad_samples, all_stars)


X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


text_input = "The salad here is pretty good. Dressing isn't too heavy and the portion size is good for its price, but the seasoning on the falafel and shawarma is just okay. This place is good for a casual/affordable lunch. 4 stars for the portion and overall taste but slightly underwhelming and bland seasoning of the meats."

print (get_feature_importance(text_input, vect, X_train_dtm, y_train).tail(50))

nb = MultinomialNB()
clf = nb.fit(X_train_dtm, y_train)

# y_pred_class = nb.predict(X_test_dtm)
# metrics.accuracy_score(y_test, y_pred_class)

    
# print ("Features: ", X_train_dtm.shape[1])
# print ("Training Score: ", nb.score(X_train_dtm, y_train))
# print ("Testing Score: ", nb.score(X_test_dtm, y_test))

# cm = confusion_matrix(y_test, y_pred_class)
# print (cm)

# #Derive probabilities of class 1 from the test set
# test_probs = nb.predict_proba(X_test_dtm)[:,1]
# #Pass in the test_probs variable and the true test labels aka y_test in the roc_curve function
# fpr, tpr, thres = metrics.roc_curve(y_test, test_probs)
# #Plotting False Positive Rates vs the True Positive Rates
# #Dotted line represents a useless model
# plt.figure(figsize=(10,8))
# plt.plot(fpr, tpr, linewidth= 8)
# #Line of randomness
# plt.plot([0,1], [0,1], "--", alpha=.7)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.show()




######## EXPORT MODEL FILES ########


with open(cwd+'/model_files/{}classifier'.format(VERSION), 'wb') as picklefile:
    pickle.dump(clf, picklefile)


with open(cwd+'/model_files/vect', 'wb') as picklefile:
    pickle.dump(vect, picklefile)



print ("Created CLF model")





