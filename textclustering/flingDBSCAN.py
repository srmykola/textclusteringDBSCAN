import matplotlib as mpl
from imp import reload
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk,re,pprint
import sys,glob,os
import operator, string, argparse, math, random, statistics
import matplotlib.pyplot as plt

class flingDBSCAN:
    def __init__(self,data,epsilon,minPts,method, metric, progress: bool = False):
        self.data = data
        self.method = method
        self.metric = metric
        self.progress = progress
        self.minPts = minPts
        self.noisePts = []
        self.nDocs = len(self.data)
        #self.clusterLabel = ['a','b','c','d','e']
        self.clusterIndex = 0 
        self.clusterCount = 0 
        print("\nflingDBSCAN initialized!\n")
        self.clusterMetadata = {}
        for i in range(self.nDocs):
            self.clusterMetadata[i] = 'no_cluster'
        if epsilon:
            self.epsilon = epsilon
        else:
<<<<<<< Updated upstream
            if method == 'glove':
                self.epsilon = self.getBestDistance('glove')
                print("\nBest epsilon computed on GLOVE =",self.epsilon,"\n")
            else:
                self.epsilon = self.getBestDistance('tfidf')
                print("\nBest epsilon computed on GLOVE-TFIDF =",self.epsilon,"\n")
            
    def getBestDistance(self,method):
=======
            self.setBestDistance()

    def setBestDistance(self):
>>>>>>> Stashed changes
        numx = 100
        numHalf = int(numx/2)
        doca,docb = [],[]
        print("computing best distance")

        for i in range(numHalf):
            doca.append(random.randint(1,int(self.nDocs/2)))
            docb.append(random.randint(int(self.nDocs/2)+1,self.nDocs))
        distanceSample = []
        total = numHalf*numHalf

        for doc_1 in range(len(doca)):
            for doc_2 in range(len(docb)):
<<<<<<< Updated upstream
                if method == 'glove':
                    distanceSample.append(self.getDistance(doc_1,doc_2,'glove'))
                else:
                    distanceSample.append(self.getDistance(doc_1,doc_2,'tfidf'))
=======
                distanceSample.append(self.getDistance(doc_1,doc_2,self.method))
>>>>>>> Stashed changes
                cov = doc_1*numHalf + doc_2
                prog=(cov+1)/total
                self.drawProgressBar(prog)

        if self.progress:
            plt.show(plt.hist(distanceSample,bins=20))

        self.epsilon = statistics.mean(distanceSample)
        print(f"\nBest epsilon computed on {self.method} = {self.epsilon}\n")

    def assignLabel(self,dictDist,label):
        for el in dictDist:
            self.clusterMetadata[el]=label
            
    def printClusterInfo(self):
        print("Cluster characteristics:")
        print(" -- vectors:",self.method)
        print(" -- minPts:",self.minPts)
        print(" -- EstimatedBestDistance",self.epsilon)
        print(" --",self.clusterCount,"clusters formed!")
        print(" --",self.nDocs-len(self.noisePts),"points assigned to clusters!") 
        print(" --",len(self.noisePts),"noise points!\n")
        noisePc = len(self.noisePts)/self.nDocs*100
        print(" --",noisePc,"% noise!\n")
            
    def printClusterMetadata(self,n):
        for j in range(n):
            print(j, self.clusterMetadata[j])
         
    # range query equivalent function
    def findNeighborOf(self,ptIndex,method):
        distance = {}
        
        #first vector
        if method == 'glove':
            dv_1 = self.data['glove-vector'][int(ptIndex)] 
        elif method == 'tfidf':
            dv_1 = self.data['tfidf2vec-tfidf'][int(ptIndex)]
        
        #iterating over the whole data for the second vector 
        if method == 'tfidf':
            for j in range(self.nDocs):
                dv_2 = self.data['tfidf2vec-tfidf'][j]
                if j!=ptIndex:
                    distx = self.getDistance(ptIndex,j,'tfidf')
                    distance[j] = distx
        elif method == 'glove':
            for j in range(self.nDocs):
                dv_2 = self.data['glove-vector'][j]
                if j!=ptIndex:
                    distx = self.getDistance(ptIndex,j,'glove')
                    distance[j] = distx
        
        # keeping only elements at a distnce of less than epsilon (or more for the cosine metric)
        if self.metric == 'cosine':
            tempDistances = {key: value for (key, value) in distance.items() if value > self.epsilon}
        else:
            tempDistances = {key:value for (key,value) in distance.items() if value<self.epsilon}

        newDistances = {key:value for (key,value) in tempDistances.items() if self.clusterMetadata[key]=='no_cluster'}
        # keeping the cluster only if we 
        if len(newDistances)>self.minPts:    
            return list(newDistances.keys())
        else:
            return None
            
    def dbscanCompute(self):
        print("\ninitiating DBSCAN Clustering with",self.method,"vectors\n")
        self.clusterMetadata[0]='cluster_0'
        for k in range(self.nDocs):
            if self.clusterMetadata[k] == 'no_cluster':
                if self.method=='glove':
                    neighbors = self.findNeighborOf(k,'glove')
                else:
                    neighbors = self.findNeighborOf(k,'tfidf')
                if neighbors:
                    self.clusterCount+=1
                    clusterName = f'cluster_{self.clusterCount}'
                    self.clusterMetadata[k] = clusterName
                    
                    # neighboring points of original point
                    for nbPoint in neighbors:
                        if self.clusterMetadata[nbPoint] == 'no_cluster':
                            self.clusterMetadata[nbPoint] = clusterName
                    if self.method=='glove':
                        innerNeighbors = self.findNeighborOf(k,'glove')
                    else:
                        innerNeighbors = self.findNeighborOf(k,'tfidf')
                    if innerNeighbors:
                        for nb in innerNeighbors:
                            self.clusterMetadata[nb] = clusterName
                            neighbors.append(nb)
                    if self.progress:
                        print("\n ---- ",clusterName,"assigned to",len(neighbors),"points! ----")
                else:
                    self.noisePts.append(k)
            prog=(k+1)/self.nDocs
            if self.progress:
                self.drawProgressBar(prog)
        print("\n",self.clusterCount,"clusters formed!")

            
    def getDistance(self,docId_1,docId_2,method):
        if method == 'glove':
            dv_1 = self.data['glove-vector'][int(docId_1)]
            dv_2 = self.data['glove-vector'][int(docId_2)]
        elif method == 'tfidf':
            dv_1 = self.data['tfidf2vec-tfidf'][int(docId_1)]
<<<<<<< Updated upstream
            dv_2 = self.data['tfidf2vec-tfidf'][int(docId_2)]           
        dist = np.linalg.norm(dv_1-dv_2)
        return dist
=======
            dv_2 = self.data['tfidf2vec-tfidf'][int(docId_2)]
        elif method == 'transformer':
            dv_1 = self.data['transformer_vector'][int(docId_1)]
            dv_2 = self.data['transformer_vector'][int(docId_2)]
            if self.metric == 'cosine':
                return np.dot(dv_1, dv_2) / (np.linalg.norm(dv_1) * np.linalg.norm(dv_2))

        return np.linalg.norm(dv_1-dv_2)
>>>>>>> Stashed changes
    
    def addClusterLabel(self,label):
        vec = []
        for el in self.clusterMetadata.keys():
            vec.append(self.clusterMetadata[el])
        self.data[label] = vec    
        
    def drawProgressBar(self, percent, barLen = 50):			#just a progress bar so that you dont lose patience
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i<int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()	
