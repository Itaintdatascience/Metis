import os
import pickle
import numpy as np
# import base64
import pandas as pd
import streamlit as st
# from bson import ObjectId
from textblob import TextBlob
# from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# https://scikit-learn.org/stable/modules/naive_bayes.html
# % cd ~/Documents/GitHub/Project_Proposal/Metis


# Get the current working directory
cwd = os.getcwd()

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    # page_icon="🧊",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    # primaryColor="purple"
)

@st.cache(suppress_st_warning=True)
def load_classifier():
    file_name = sorted([k for k in os.listdir(cwd+'/model_files/') if "classifier" in k])
    path = cwd+'/model_files/{}'.format(file_name[0])
    f = open(path, 'rb')
    clf = pickle.load(f)
    f.close()
    return clf

@st.cache(suppress_st_warning=True)
def load_vect():
    # define a function that accepts text and returns a list of lemons (verb version)
    f = open(cwd+'/model_files/vect', 'rb')
    vect = pickle.load(f)
    f.close()
    return vect


def word_tokenize_lemma_verb(text):
    words = TextBlob(text).words
    return [word.lemmatize(pos = "v") for word in words]


vect = CountVectorizer(
                  stop_words="english", 
                  analyzer=word_tokenize_lemma_verb,
                  max_features=10000
            )


def load_feat_importance():
    featureImportance = pd.read_csv(cwd+'/featureImportance.csv')
    return featureImportance



# @st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# @st.cache
def load_reviews():
    randomize = st.checkbox('Show Examples', value = False)
    
    if randomize:
        col1, col2 = st.columns((1,1))
        a = col1.write("Bad Review Example: ")
        b = col2.write("Good Review Example: ")
        aa = col1.write(df_rw[df_rw.target == 0]['text'].sample(n = 1).values[0])
        bb = col2.write(df_rw[df_rw.target == 1]['text'].sample(n = 1).values[0])

        return a, aa, b, bb


# PLOT MODEL PERFORMANCE
def build_model(model, vect):
    pick_model = st.checkbox('pick model', value = False)
    
    if pick_model:
        # Model Summary:
        X_test_dtm = vect.transform(X_test)
        y_pred_class = model.predict(X_test_dtm)
        st.write("Testing Score: ", model.score(X_test_dtm, y_test))
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix(y_test, y_pred_class))

        # PLOTS for Test
        col1, col2 = st.columns((1,1))
        #Derive probabilities of class 1 from the test set
        test_probs = model.predict_proba(X_test_dtm)[:,1]
        #Pass in the test_probs variable and the true test labels aka y_test in the roc_curve function
        fpr, tpr, thres = metrics.roc_curve(y_test, test_probs)
        #Plotting False Positive Rates vs the True Positive Rates
        #Dotted line represents a useless model
        fig1, ax = plt.subplots()
        plt.plot(fpr, tpr, linewidth= 5, c='c', alpha= 0.5)
        #Line of randomness
        plt.plot([0,1], [0,1], "--", alpha=.5, c='m')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        col1.pyplot(fig1, figsize=(1.5, 2.5))

        fig2, ax = plt.subplots()
        plt.plot(thres, fpr, linewidth=5, label = "FPR Line", alpha=0.5, c='m')
        plt.plot(thres, tpr, linewidth=5, label = "TPR line", alpha=0.5, c='c')
        plt.xlabel("Thresholds")
        plt.ylabel("False Positive Rate")
        plt.legend()
        col2.pyplot(fig2, figsize=(1.5, 1.5))


        #Caculate the area under the curve score using roc_auc_score using SKLEARN. only work with Binary Classification
        auc_score = roc_auc_score(y_test, test_probs)
        st.write("The Area Under the Curve (AUC) score is: `{}`".format(auc_score))

        #Cross validated roc_auc score
        cv_score = cross_validate(clf, y_train, cv=5, scoring="roc_auc")
        st.write("The Cross Validated (at {} folds) Area Under the Curve (AUC) score is: `{}`".format(5, cv_score['test_score'].mean()))


@st.cache(suppress_st_warning=True)
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


def get_good_features(featureImportance, PAYLOAD_TEXT, VECT):
    payload_features = pd.DataFrame(VECT.transform([PAYLOAD_TEXT]).toarray(), columns=VECT.get_feature_names(), index=['values']).T
    payload_features = payload_features[payload_features.values>0].reset_index()    
    payload_features.columns = ['tokens', 'values']
    good_features = featureImportance.merge(payload_features['tokens'], on="tokens", how='inner').sort_values('featureImportance', ascending=False)

    return good_features


def get_feat_text(slim, list_string):
    list_a = []

    for i in list_string:

        if i in slim:
            p = "`"+str(i)+"`"
            list_a.append(p)
        else:
            list_a.append(i)

    out = " ".join(list_a) 
    return out




col1, col2 = st.columns((1,5))
col1.image(cwd+'/yelp_burst.png', width=100)
col2.write(
'''
# Yelp Star Rating Classifier - Yelp Dataset''')

st.write(
'''
Maybe you've heard about a restaurant that you've been wanting to try. You've heard comments from friends and family, but don't quite trust their opinion! 
Given a sample of text, let's predict if this will be a good or bad review. 


[Additional info on the dataset](https://yelp.com/dataset/)
'''
)







# clf is a nb classifier. Load pre-trained model files:
clf = load_classifier()
vect = load_vect()
featureImportance = load_feat_importance()
featureImportance.columns = ['index','tokens', 'featureImportance']
featureImportance = featureImportance.drop('index', axis=1)


st.write(
'''
### Predict if this is a Good or Bad review:
'''
)

###### CREATE TEXT INPUT FIELD FOR PAYLOAD TESTING ######

text_input = st.text_area("What have you heard about this place?:", height=150)

payload_transformed = vect.transform([text_input])

list_string = text_input.split(' ')
slim = [k for k in list_string if k in list(featureImportance['tokens'])]



output = clf.predict(payload_transformed)[0]
proba = clf.predict_proba(payload_transformed)

st.text('')
st.text('')
st.text('')
st.text('')

col1, col2 = st.columns((1,1))
col1.write(
'''
##### Probability of a Good or Bad Review:
'''
)
col1.dataframe(pd.DataFrame(clf.predict_proba(payload_transformed), columns = ['prob_Bad', 'prob_Good']))


if text_input:
    if output == 1:
        output_str = "Good Review"
        col2.write("This is a `{}`!".format(output_str))
    else:
        output_str = "Bad Review"
        col2.write("This is a `{}`.. 🧐".format(output_str))



st.text('')
st.text('')
st.text('')
st.text('')



# Sentiment Analysis
tooltip_text = "`Polarity` measures how happy or mad or a text is on a scale from -1.0 to 1.0. `Subjectivity` measures how strongly opinonated a text is on a scale from 0.0 to 1.0."

df_sentiment = pd.DataFrame(TextBlob(text_input).sentiment).T
df_sentiment.columns = ['polarity','subjectivity']
df_sentiment = df_sentiment.T
df_sentiment.columns = ['value']
df_sentiment = df_sentiment.reset_index()

list_a = []
for ix, k in df_sentiment.iterrows():

    if (k['index'] == "polarity") & (k['value'] < 0):
        list_a.append('negative')
    elif (k['index'] == "subjectivity") & (k['value'] <= 0.5):
        list_a.append('negative')
    else:
        list_a.append('positive')

df_sentiment["Vibe"] =  list_a


# # Plot
fig = px.bar(        
        df_sentiment,
        y = 'value',
        title = "Sentiment Analysis",
        text_auto=True,
        color='Vibe',   
        color_discrete_map={
            'negative': 'salmon',
            'positive': 'turquoise'
        },
        pattern_shape="value", pattern_shape_sequence=[".", "+"]
    )



fig.update_traces(textposition="outside")
fig.update_layout(showlegend=False)
fig.update_yaxes(range = [-1,1], ticks="outside", tickwidth=2, tickcolor='crimson')

col1, col2 = st.columns((1,1))


col1.plotly_chart(fig)
col2.text('')
col2.text('')
col2.text('')
col2.text('')
col2.text('')
col2.text('')


col2.write(tooltip_text)

if TextBlob(text_input).subjectivity >= 0.5:
    subjectivity = "opinionated"
else:
    subjectivity = "not opinionated"

if TextBlob(text_input).polarity <= 0:
    col2.write("This review has `{}` and review is `{}`".format("Negative Vibes", subjectivity))

else:
    col2.write("This review has `{}` and review is `{}`".format("Positive Vibes", subjectivity))



col1, col2 = st.columns((1,1))
good_features = get_good_features(featureImportance, text_input, vect)
col1.write("Here are the features of importance:")

if output == 1:

    col1.dataframe(good_features.sort_values('featureImportance', ascending=False))

else:
    col1.dataframe(good_features.sort_values('featureImportance', ascending=True))


col2.text('')
col2.text('')
col2.caption("Highlighted text are of Feature Importance from the model:")
col2.write(get_feat_text(slim, list_string))






