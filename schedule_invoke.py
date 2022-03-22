# scheduler
from schedule import every, repeat, run_pending
from pymongo import MongoClient
from textblob import TextBlob
import base64
from bson.objectid import ObjectId
import pickle
import time
import os

cwd = os.getcwd()

# import yelp dataset to mongoDB server
#  mongoimport ~/Downloads/archive/yelp_academic_dataset_review.json -d yelpdb_invoke -c checkin --drop

client = MongoClient()
db = client[ "testdb" ] # makes a test database called "testdb"
col = db[ "testcol" ]

def word_tokenize_lemma_verb(text):
    words = TextBlob(text).words
    words = [word.lemmatize(pos = "v") for word in words]
    return words

def load_classifier():
    path = cwd+'/model_files/nb_classifier_1'
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

@repeat(every(2).seconds)
def job():
    print("I am a scheduled job")
    """
    Create MongoDB scheduler to refresh data payload to be scored with latest version of classifier model and vectorizer:
    """
    # set MongoDB:
    # client = MongoClient()
    # db_invoke = client.yelpdb_invoke
    # collection_invoke = db_invoke.preds
    
    payload = "Stopped by and got the pepperoni fat slice (2 big slices for approx $15) and was fairly disappointed. The crust was super hard and the overall flavor was subpar. I ended up eating only a single slice and threw the other slice away, something I have never done. It's not that it had a terrible flavor but everything about the pizza was lackluster and uninspiring.Maybe it was an off day or they are dealing with some issues bc of covid? If this was a typical night, I would definitely not recommend. It may be a long time if I ever decide to stop by again for some pizza here :("
    pred = clf.predict(vect.transform([payload]))[0]
    proba_bad, proba_good = clf.predict_proba(vect.transform([payload]))[0]
    mydict = { "text": payload, "proba_bad": proba_bad, "proba_good": proba_good, "pred": pred, "model_version": version}
    mydict["_id"] = str(ObjectId())
    mydict["_id"] = str(ObjectId())
    collection_invoke.insert(mydict)

while True:
    run_pending()
    time.sleep(1)






