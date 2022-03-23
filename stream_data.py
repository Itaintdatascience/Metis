# Stream new data into reviews collection in MongoDB
from schedule import every, repeat, run_pending
from pymongo import MongoClient
import pymongo
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
    
    docs = list(db.reviews.aggregate([{'$sample': {'size': 5000}}]))
    # db.prod.insert_many(docs)
    count=0
    for i in docs:

        try:
            db.reviews.delete_one(i)
            db.prod.insert_one(i)

        except pymongo.errors.DuplicateKeyError:
            continue

            count+=1

    print ('streamed {} docs from `reviews` to `prod` collection'.format(count))

while True:
    run_pending()
    time.sleep(1)