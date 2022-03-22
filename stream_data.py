# Stream new data into reviews collection in MongoDB
from schedule import every, repeat, run_pending
from pymongo import MongoClient
import time

client = MongoClient()

db = client.yelpdb
collection_prod = db.prod
# collection_reviews = db_yelp.reviews
# create new collection for gathering data



@repeat(every(30).seconds)
def job():
    """
    Create MongoDB scheduler to stream data from `reviews` to `production` level data
    """
    docs = list(db.reviews.aggregate([{'$sample': {'size': 10000}}]))
    db.prod.insert(docs)
    # for i in docs: 
    db.reviews.delete_many(docs)

    # mydict = { "text": payload, "stars": doc[0]['stars'], "model_version": str(version), "proba_bad": str(proba_bad), "proba_good": str(proba_good), "pred": str(pred)}
    # mydict["_id"] = str(ObjectId())
    # # add to results collection
    # col_results.insert_one(mydict)
    # # remove from invoke list
    # print (mydict)
    # collection_invoke.delete_one(doc[0])

while True:
    run_pending()
    time.sleep(1)