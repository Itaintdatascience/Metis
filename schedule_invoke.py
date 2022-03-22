# scheduler
from schedule import every, repeat, run_pending
from pymongo import MongoClient
from textblob import TextBlob
import base64
from os import listdir
from bson.objectid import ObjectId
import pickle
import time
import os
import json

cwd = os.getcwd()
client = MongoClient()
# import yelp dataset to mongoDB server
#  mongoimport ~/Downloads/archive/yelp_academic_dataset_review.json -d yelpdb_invoke -c checkin --drop

db_invoke = client.yelpdb
collection_invoke = db_invoke.invoke_payloads

col_results = db_invoke.results


def word_tokenize_lemma_verb(text):
    words = TextBlob(text).words
    words = [word.lemmatize(pos = "v") for word in words]
    return words

def load_classifier():
    file_name = [k for k in os.listdir(cwd+'/model_files/') if "classifier" in k]
    path = cwd+'/model_files/{}'.format(file_name)
    f = open(path, 'rb')
    clf = pickle.load(f)
    f.close()
    return clf, path[-1]

def load_vect():
    # define a function that accepts text and returns a list of lemons (verb version)
    f = open(cwd+'/model_files/vect', 'rb')
    vect = pickle.load(f)
    f.close()
    return vect


clf, version = load_classifier()
vect = load_vect()



@repeat(every(10).seconds)
def job():
    """
    Create MongoDB scheduler to refresh data payload to be scored with latest version of classifier model and vectorizer:
    """
    # set MongoDB:
    # client = MongoClient()

    print("I am a scheduled job")
    # read in payload:
    doc = list(collection_invoke.find().limit(1))
    # print (doc)

    payload = doc[0]['text']
    pred = clf.predict(vect.transform([payload]))[0]
    proba_bad, proba_good = clf.predict_proba(vect.transform([payload]))[0]
    mydict = { "text": payload, "stars": doc[0]['stars'], "model_version": str(version), "proba_bad": str(proba_bad), "proba_good": str(proba_good), "pred": str(pred)}
    mydict["_id"] = str(ObjectId())
    # add to results collection
    col_results.insert_one(mydict)
    # remove from invoke list
    collection_invoke.delete_one(doc[0])

while True:
    run_pending()
    time.sleep(1)






# moving files from one collection to the other, then deleting documents
# docs = list(db.reviews.find().limit(10000))
# db.test.insert(docs)

# for i in docs: 
#     db.reviews.remove(i)



