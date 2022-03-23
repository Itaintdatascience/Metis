import os
import pickle
import numpy as np
import base64
import pandas as pd
import streamlit as st
from bson import ObjectId
from textblob import TextBlob
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# import requests
import matplotlib.pyplot as plt
# https://scikit-learn.org/stable/modules/naive_bayes.html
# % cd ~/Documents/GitHub/Project_Proposal/Metis

# Get the current working directory
cwd = os.getcwd()

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    # page_icon="🧊",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_classifier():
    f = open(cwd+'/model_files/1_nb_classifier', 'rb')
    clf = pickle.load(f)
    f.close()
    return clf


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


def download_model(model):

    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    if model == clf:
        href = f'<a href="data:file/output_model;base64,{b64}">Download_clf.pkl</a>'
    elif model == vect:
        href = f'<a href="data:file/output_model;base64,{b64}">Download_vect.pkl</a>'
    st.markdown(href, unsafe_allow_html=True)


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')





st.write(
'''
# Data Engineering (March 2022)
### *Yelp Star Rating Classifier - Yelp Dataset*
Maybe you've heard about a restaurant that you've been wanting to try. You've heard comments from friends and family, but don't quite trust their opinion! 
Given a sample of text, we try to predict if this will be a good or bad review. 
Sample the out of the box model that is already pre-trained, or learn how to build a text classifier model on your own!

[Additional info on the dataset](https://yelp.com/dataset/)

'''
)


# Connect to MongoDB
client = MongoClient()
# client.list_database_names()
db = client.yelpdb
collection = db.prod


# GLOBALS
n_good_bad_samples = 25000
all_stars = [1.0,2.0,3.0,4.0,5.0]
# good and bad star ratings for creating target variable
BAD = [1.0,2.0,3.0]
GOOD = [4.0,5.0]

@st.cache()
def load_data(NUM_REVIEWS_EACH, STARS):
    df_rw_good = pd.DataFrame(list(collection.find({"stars":{"$in":[k for k in STARS if k in GOOD]}}, {"text": 1, 'stars':1}).limit(NUM_REVIEWS_EACH)))
    # st.write([k for k in STARS if k in BAD])
    df_rw_bad = pd.DataFrame(list(collection.find({"stars":{"$in":[k for k in STARS if k in BAD]}}, {"text": 1, 'stars':1}).limit(NUM_REVIEWS_EACH)))
    df_rw = pd.concat([df_rw_bad, df_rw_good], axis=0)
    df_rw.reset_index(inplace=True, drop=True)
    del df_rw_bad
    del df_rw_good
    df_rw = df_rw.astype(str)
    # if len(STARS) < 5:
    #     df_rw = df_rw[df_rw.stars.isin[STARS]]

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




use_example_model = st.checkbox(
    "Use example model", True, help="Use pre-built example model to demo the app"
)


# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
if use_example_model:

    # clf is a nb classifier. Load pre-trained model files:
    clf = load_classifier()
    vect = load_vect()

    df_rw, X_train, X_test, y_train, y_test, X, y = load_data(n_good_bad_samples, all_stars)

    # plot learning curve
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    col1, col2 = st.columns((1,1))

    col1.write("Examples of Bad Reviews: ")
    col1.write(df_rw[df_rw.target == 0]['text'].sample(n = 1).values[0])
    col2.write("Examples of Good Reviews: ")
    col2.write(df_rw[df_rw.target == 1]['text'].sample(n = 1).values[0])

    st.write(
    '''
    ## Predict if this is a Good or Bad review:
    '''
    )

    ###### CREATE TEXT INPUT FIELD FOR PAYLOAD TESTING ######

    text_input = st.text_input("What have you heard about this place?:")
    st.write(text_input)
    payload_transformed = vect.transform([text_input])

    output = clf.predict(payload_transformed)[0]
    proba = clf.predict_proba(payload_transformed)
    st.write("Probability:")
    st.write(pd.DataFrame(clf.predict_proba(payload_transformed), columns = ['prob_Bad', 'prob_Good']))

    if text_input:
        if output == 1:
            output_str = "Good Review"
            st.write("This is a `{}`!".format(output_str))
        else:
            output_str = "Bad Review"
            st.write("This is a `{}`.. 🧐".format(output_str))



    if text_input:
        # Do you Agree with the review?
        # Collect Text Input
        client = MongoClient()
        db_feedback = client.yelpdb_feedback
        collection_feedback = db_feedback.reviews

        mydict = { "text": text_input, "target": output.astype(str)}

        x = collection_feedback.insert_one(mydict)
        x_id = x.inserted_id

        # GATHER FEEDBACK. Update document if the "target" is incorrect per user feedback!
        st.write("Do you think the prediction is correct? If not, please provide feedback: ")
        # FEEDBACK = (0, 1) dropdown bar (0 for "Bad Review", 1 for "Good Review")
        feedback = st.selectbox('Pick one', ['--', 'Bad Review', 'Good Review'])
        
        if feedback == "Bad Review":
            doc = collection_feedback.find_one_and_update(
            {"text" : text_input, "_id" : ObjectId(x_id)},
            {"$set":
                {"target": 0, "proba_bad" : proba[0][0], "proba_good" : proba[0][1]}
            },upsert=True
            )
            st.write('Thank you for your feedback. We will consider this in the next model!')

        elif feedback == "Good Review":
            doc = collection_feedback.find_one_and_update(
            {"text" : text_input, "_id" : ObjectId(x_id)},
            {"$set":
                {"target": 1}
            },upsert=True
            )
            st.write('Thank you for your feedback. We will consider this in the next model!')

        else:
            pass


        X_train_dtm = vect.fit_transform(X_train)
        # st.dataframe(get_feature_importance(text_input, vect, X_train_dtm, y_train))
    

        if output_str == "Bad Review":
            st.dataframe(get_feature_importance(text_input, vect, X_train_dtm, y_train).tail(50))

        else:
            st.dataframe(get_feature_importance(text_input, vect, X_train_dtm, y_train).head(50))



# BUILD YOUR OWN MODEL SECTION:

else:

    # # Sidebar items:
    st.sidebar.markdown("# Controls")

    n_good_bad_samples = st.sidebar.number_input('Select number of samples for each group:', 1, 200000, 50000)
    pick_model = st.sidebar.selectbox(
    "Pick a Classifier Model: ",
    ("MultinomialNB", "Logistic_Regression"))   

    # include star ratings for train/test split
    stars = st.sidebar.multiselect(
        "Stars", options=all_stars, default=all_stars
    )

    def add_parameter(clf_name):

        params = {}
        if pick_model == "MultinomialNB":
            max_features = st.sidebar.slider("max_features", min_value=1, max_value=20000, value=10000)
            params["max_features"] = max_features
        elif pick_model == "Logistic_Regression":
            max_features = st.sidebar.slider("max_features", min_value=1, max_value=20000, value=10000)
            params["max_features"] = max_features
    
        return params

    df_rw, X_train, X_test, y_train, y_test, X, y = load_data(n_good_bad_samples, stars)

    params = add_parameter(pick_model)


    lower = st.sidebar.checkbox(
        "Lowercase", False, help="Lowercase tokens before training"
        )

    if lower:
        params['lowercase'] = True
    else:
        params['lowercase'] = False

    bigram = st.sidebar.checkbox(
        "Use Bigrams", False, help="Enable ngram feature: (1,2)"
        )
    trigram = st.sidebar.checkbox(
        "Use Trigrams", False, help="Enable ngram feature: (1,3)"
        )
    if bigram:
        params['ngram_range'] = (1,2)
    elif trigram:
        params['ngram_range'] = (1,3)
    else:
        params['ngram_range'] = (1,1)



    vect = CountVectorizer(
                  # stop_words="english", 
                  # analyzer=word_tokenize_lemma_verb,
                  # tokenizer = LemmaTokenizer(),
                  lowercase = params['lowercase'],
                  max_features=params['max_features'],
                  ngram_range=params['ngram_range']

            )


    def build_model(model, CLF, vect):
        # Model Summary:
        st.write("Selected Model: `{}`".format(pick_model))
        y_pred_class = model.predict(X_test_dtm)
        st.write("Dataset shape: `{}`".format(df_rw.shape))
        st.write("Train set shape: `{}`".format(X_train_dtm.shape))
        st.write("Test set shape: `{}`".format(X_test_dtm.shape))
        st.write("Training Score: ", model.score(X_train_dtm, y_train))
        st.write("Testing Score: ", model.score(X_test_dtm, y_test))
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix(y_test, y_pred_class))

        # PLOTS for Test
        col1, col2 = st.columns((1,1))
        #Derive probabilities of class 1 from the test set
        test_probs = clf.predict_proba(X_test_dtm)[:,1]
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
        col2.pyplot(fig2, figsize=(1.5, 2.5))


        #Caculate the area under the curve score using roc_auc_score using SKLEARN. only work with Binary Classification
        auc_score = roc_auc_score(y_test, test_probs)
        st.write("The Area Under the Curve (AUC) score is: `{}`".format(auc_score))

        folds = st.sidebar.slider("Select nfolds for Cross Validation:", 2, 10, 5)
        #Cross validated roc_auc score
        cv_score = cross_validate(clf, X_train_dtm, y_train, cv=folds, scoring="roc_auc")
        st.write("The Cross Validated (at {} folds) Area Under the Curve (AUC) score is: `{}`".format(folds,cv_score['test_score'].mean()))

        # Download pickle files for clf and vect:
        st.write("Export model files:")
        download_model(clf)
        download_model(vect)

        # Build a dataframe with mis-classified items for feedback loop
        misses = pd.DataFrame(X_test)
        misses['proba_bad'] = clf.predict_proba(X_test_dtm)[:,0]
        misses['proba_good'] = clf.predict_proba(X_test_dtm)[:,1]
        misses['pred'] = y_pred_class
        misses['actual'] = y_test
        list_ix = []
        for ix, k in misses.iterrows():
            if k['actual'] != k['pred']:
                list_ix.append(ix)
        list_ix = list(set(list_ix))
        test1 = misses[misses.index.isin(list_ix)]
        test1.sort_index(inplace=True)
        test2 = df_rw[df_rw.index.isin(list_ix)]
        csv = convert_df(test1.merge(test2.drop('target', axis=1), how='inner'))

        st.write("Export file of mis-classified predictions for re-training:")
        
        st.download_button(
             label="Download all mis-classified",
             data=csv,
             file_name='misses_feedback.csv',
             mime='text/csv',
        )
        st.write(
        """
        Here are the value counts of stars mis-classified. (*FP and FN, split by star rating*))
        """
        )
        st.write(test2.stars.value_counts(normalize=False))


        st.write(
        '''
        ### Predict if this is a Good or Bad review as payload test:
        '''
        )

        ###### CREATE TEXT INPUT FIELD FOR PAYLOAD TESTING ######
        text_input = st.text_input("What have you heard about this place?:")
        st.write(text_input)
        payload_transformed = vect.transform([text_input])

        output = clf.predict(payload_transformed)[0]
        proba = clf.predict_proba(payload_transformed)
        st.write("Probability:")
        st.write(pd.DataFrame(clf.predict_proba(payload_transformed), columns = ['prob_Bad', 'prob_Good']))

        if text_input:
            if output == 1:
                output_str = "Good Review"
                st.write("This is a `{}`!".format(output_str))
            else:
                output_str = "Bad Review"
                st.write("This is a `{}`.. 🧐".format(output_str))

        if text_input:
            # Do you Agree with the review?
            # Collect Text Input
            client = MongoClient()
            db_feedback = client.yelpdb_feedback
            collection_feedback = db_feedback.reviews

            mydict = { "text": text_input, "target": output.astype(str)}

            x = collection_feedback.insert_one(mydict)
            x_id = x.inserted_id

            # GATHER FEEDBACK. Update document if the "target" is incorrect per user feedback!
            st.write("Do you think the prediction is correct? If not, please provide feedback: ")
            # FEEDBACK = (0, 1) dropdown bar (0 for "Bad Review", 1 for "Good Review")
            feedback = st.selectbox('Pick one', ['--', 'Bad Review', 'Good Review'])
            
            if feedback == "Bad Review":
                doc = collection_feedback.find_one_and_update(
                {"text" : text_input, "_id" : ObjectId(x_id)},
                {"$set":
                    {"target": 0, "proba_bad" : proba[0][0], "proba_good" : proba[0][1]}
                },upsert=True
                )
                st.write('Thank you for your feedback. We will consider this in the next model!')

            elif feedback == "Good Review":
                doc = collection_feedback.find_one_and_update(
                {"text" : text_input, "_id" : ObjectId(x_id)},
                {"$set":
                    {"target": 1}
                },upsert=True
                )
                st.write('Thank you for your feedback. We will consider this in the next model!')

            else:
                pass
            # Feature Importance
            if output_str == "Bad Review":

                st.dataframe(get_feature_importance(text_input, vect, X_train_dtm, y_train).tail(50))
            else:

                st.dataframe(get_feature_importance(text_input, vect, X_train_dtm, y_train).head(50))





    
    if pick_model == "MultinomialNB":
        nb = MultinomialNB()
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        clf = nb.fit(X_train_dtm, y_train)
            # Build custom NB model:
        build_nb = build_model(nb, clf, vect)




    elif pick_model == "Logistic_Regression":
        lr = LogisticRegression()
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        clf = lr.fit(X_train_dtm, y_train)
        build_lr = build_model(lr, clf, vect)


        

    # elif pick_model == "Spark_Classifier":
    #     data = df_rw
    #     train, test = data.randomSplit([0.7, 0.3], seed = 42)





        # CONNECT TO STARS in SIDEBAR
        # plot_df = _df[_df.lang.isin(langs)]
        # plot_df["stars"] = plot_df.stars.divide(1000).round(1)

        # chart = (
        #     alt.Chart(
        #         plot_df,
        #         title="Static site generators popularity",
        #     )
        #     .mark_bar()
        #     .encode(
        #         x=alt.X("stars", title="'000 stars on Github"),
        #         y=alt.Y(
        #             "name",
        #             sort=alt.EncodingSortField(field="stars", order="descending"),
        #             title="",
        #         ),
        #         color=alt.Color(
        #             "lang",
        #             legend=alt.Legend(title="Language"),
        #             scale=alt.Scale(scheme="category10"),
        #         ),
        #         tooltip=["name", "stars", "lang"],
        #     )
        # )


        # st.altair_chart(chart, use_container_width=True)








    # st.write(
    # '''
    # ### How to adjust model parameters to get even smarter...



    # 2nd LEVEL ADD FEEDBACK LOOP.. 
    # This was a miss. how to re-learn (what was wrong about this?)

    # "i love the everything but the kitchen sink pizza"..


    # Create an auto-update ID for each unique row to append
    # check in MongoDB

    # User imput of columns
    # - read if you already have a row for this, if not.. then add in a simple "update feedback" 

    # - unique text comment vs. text id..

    # - Feature Importance (NB)??? What does it look like for the Text Input?
    # https://blog.ineuron.ai/Feature-Importance-in-Naive-Bayes-Classifiers-5qob5d5sFW#:~:text=The%20naive%20bayes%20classifers%20don,class%20with%20the%20highest%20probability.

    # '''
    # )



