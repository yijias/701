import os
import numpy as np
import random

def stringToNumber(reviewList):
    reviewers = list(set([review[0] for review in reviewList]))

    #map reviewerID to order number
    reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))

    return reviewers,reviewersMap

def findValid():
    path = os.getcwd()
    file = path + '/ratings_Musical_Instruments.csv'
    f = open(file,'r')
    rawString = f.read()
    rawList = rawString.split('\n')[:-1]
    for i in range(len(rawList)):
        rawList[i] = rawList[i].split(',')
        rawList[i] = tuple(rawList[i][:-1])
    #Above got the raw list of all reviews
    reviewers, reviewersMap= stringToNumber(rawList)
    reviewerbyreviews = np.array(rawList)[:,0]
    #count the number of reviews by each person
    values, counts = np.unique(reviewerbyreviews, return_counts=True)
    validSeq = np.argwhere(counts>2)
    validList = [review for review in rawList if [reviewersMap[review[0]]] in validSeq]
    return validList



def separate(validList):
    reviewers, reviewersMap= stringToNumber(validList)
    reviewersbyreviews = [review[0] for review in validList]
    validSet = set(validList)
    testSet = set()
    for reviewer in reviewers:
        testSet.add(validList[reviewersbyreviews.index(reviewer)])
    trainSet = validSet.difference(testSet)
    return trainSet,testSet
    #Split training data and test data

    #print(reviewsbyreviewer)
    #mapping from ID to number

    #for i in range(len(rawList)):
    #    rawList[i] = tuple([reviewersMap[rawList[i][0]], productsMap[rawList[i][1]], float(rawList[i][2])])
    #print(rawList[0])


validList = findValid()
trainSet,testSet = separate()
