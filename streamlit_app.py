
"""
Streamlit Housing App Demo
    
Make sure to install Streamlit with `pip install streamlit`.

Run `streamlit hello` to get started!

To run this app:

1. cd into this directory
2. Run `streamlit run streamlit_app.py`
"""

# import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pymongo import MongoClient
# import requests
import numpy as np
import pickle
from textblob import TextBlob
# from sklearn.metrics.pairwise import cosine_similarity

# Import the os module
import os
# Get the current working directory
cwd = os.getcwd()



def load_classifier():
    f = open(cwd+'/model_files/nb_classifier', 'rb')
    clf = pickle.load(f)
    f.close()
    return clf  

def word_tokenize_lemma_verb(text):
    words = TextBlob(text).words
    return [word.lemmatize(pos = "v") for word in words]

def load_vect():
    # define a function that accepts text and returns a list of lemons (verb version)

    f = open(cwd+'/model_files/vect', 'rb')
    vect = pickle.load(f)
    f.close()
    return vect





st.write(
'''
# Data Engineering (March 2022)
### *Yelp Star Rating Classifier - Yelp Dataset*

1. Text input for guess on Star Rating!

[Additional info on the dataset](https://yelp.com/dataset/)

'''
)

client = MongoClient()
# client.list_database_names()
db = client.yelp_reviews
collection = db.yelp_reviews

# Build a sample dataframe from MongoDB


# Stream data in from MongoDB Server:

# @st.cache
# def fetch_data():
# sample = list(collection.find({}, {'text':1, 'stars':1}).limit(100000))
df_rw_good = pd.DataFrame(list(collection.find({"stars":{"$gte":4}}, {"text": 1, 'stars':1}).limit(50000)))
df_rw_bad = pd.DataFrame(list(collection.find({"stars":{"$lt":4}}, {"text": 1, 'stars':1}).limit(50000)))
df_rw = pd.concat([df_rw_bad, df_rw_good], axis=0)
st.write(df_rw.columns)
df_rw = df_rw.astype(str)



def good_bad_review(x):
    if x >= 4:
        return 1
    else:
        return 0

df_rw['stars'] = df_rw['stars'].astype(float)
df_rw.target = df_rw['stars'].apply(good_bad_review)


clf = load_classifier()
vect = load_vect()


# plot learning curve
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


st.dataframe(df_rw.head(5))



###### CREATE TEXT INPUT FIELD ######
text_input = "This place is nice. It was crowded but the line moved really fast and the food came out quickly too. It had a wide variety of items on the the menu. The food however wasn't anything special for the price and sides are not included so a burger with fries and cost upwards of $10 adding a drink and tax and it's close to $15. The patio seating area provides a nice environment to eat in but you have to watch out for the birds and bugs. All in all it was a nice place I'd eat here again if I had to but it wouldn't be my #1 choice."
text_input = st.text_input("What have you heard about this place?:")
st.write(text_input)
a = vect.transform([text_input])


st.write(
'''
### Good or Bad review?
'''
)

output = clf.predict(a)[0]
st.write("Probability:")
st.write(clf.predict_proba(a))

if output == 1:
    st.write("Good Review!")
else:
    st.write("Bad Review.. 🧐")




st.write(
'''
### How to adjust model parameters to get even smarter:
'''
)


# # Sidebar items:
# st.sidebar.markdown("# Controls") # Must be .markdown method, not .write method
# st.sidebar.markdown("Adjust start index and number of months")
# with st.sidebar:
#     add_selectbox = st.sidebar.selectbox(
#     "Pick a company: ",
#     (0, 1, 2)
# )

#Get the index values of the 3
# st.write("Selected business is {}, and the categories are {}".format(company_dict[add_selectbox], data.iloc[add_selectbox]['categories']))

# dist_column = df_dist[add_selectbox]

# # st.write(data.iloc[add_selectbox]['categories'])

# closest_index = dist_column.nlargest(5).index[1:].tolist()

# st.write(
# '''
# ## 
# Suggested businesses based on selection:
# ''')
# for i in closest_index:
#         st.write(i)
#         st.write((data.iloc[i]['name'], ": ", data.iloc[i]['categories']))
#         st.write('Cosine Similarity Score:', cosine_similarity(dist[add_selectbox].reshape(1,-1), dist[i].reshape(1,-1))[0][0])
#         st.write()



## PART 4 - Graphing and Buttons
#
# st.write(
# '''
# ### Graphing and Buttons
# Let's graph some of our data with matplotlib. We can also add buttons to add interactivity to our app.
# '''
# )

# fig, ax = plt.subplots()

# ax.hist(data['PRICE'])
# ax.set_title('Distribution of House Prices in $100,000s')

# show_graph = st.checkbox('Show Graph', value=True)

# if show_graph:
#     st.pyplot(fig)


# ## PART 5 - Mapping and Filtering Data
# #
# st.write(
# '''
# ## Mapping and Filtering Data
# We can also use Streamlit's built in mapping functionality.
# Furthermore, we can use a slider to filter for houses within a particular price range.
# '''
# )

# price_input = st.slider('House Price Filter', int(data['PRICE'].min()), int(data['PRICE'].max()), 500000 )

# price_filter = data['PRICE'] < price_input
# st.map(data.loc[price_filter, ['lat', 'lon']])


# # PART 6 - Linear Regression Model

# st.write(
# '''
# ## Train a Linear Regression Model
# Now let's create a model to predict a house's price from its square footage and number of bedrooms.
# '''
# ) 


# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# clean_data = data.dropna(subset=['PRICE', 'SQUARE FEET', 'BEDS'])

# X = clean_data[['SQUARE FEET', 'BEDS']]
# y = clean_data['PRICE']

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# ## Warning: Using the above code, the R^2 value will continue changing in the app. Remember this file is run upon every update! Set the random_state if you want consistent R^2 results.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# lr = LinearRegression()
# lr.fit(X_train, y_train)

# st.write(f'Test RÂ²: {lr.score(X_test, y_test):.3f}')


# # PART 7 - Predictions from User Input

# st.write(
# '''
# ## Model Predictions
# And finally, we can make predictions with our trained model from user input.
# '''
# )

# sqft = st.number_input('Square Footage of House', value=2000)
# beds = st.number_input('Number of Bedrooms', value=3)

# input_data = pd.DataFrame({'sqft': [sqft], 'beds': [beds]})
# pred = lr.predict(input_data)[0]

# st.write(
# f'Predicted Sales Price of House: ${int(pred):,}'
# )