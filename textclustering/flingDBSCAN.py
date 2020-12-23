import numpy as np

import sys
import random
import statistics
import matplotlib.pyplot as plt

class flingDBSCAN:
    def __init__(self, data, epsilon, minPts, method, metric, progress: bool = False):
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
            self.setBestDistance()

    def setBestDistance(self):

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
                distanceSample.append(self.getDistance(doc_1,doc_2))
                cov = doc_1*numHalf + doc_2
                prog=(cov+1)/total
                if self.progress:
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
    def findNeighborOf(self, ptIndex):
        distance = {}
        
        #first vector
        if self.method == 'glove':
            dv_1 = self.data['glove-vector'][int(ptIndex)] 
        elif self.method == 'tfidf':
            dv_1 = self.data['tfidf2vec-tfidf'][int(ptIndex)]
        elif self.method == 'transformer':
            dv_1 = self.data['transformer_vector'][int(ptIndex)]
        
        #iterating over the whole data for the second vector 
        if self.method == 'tfidf':
            for j in range(self.nDocs):
                dv_2 = self.data['tfidf2vec-tfidf'][j]
                if j!=ptIndex:
                    distx = self.getDistance(ptIndex,j)
                    distance[j] = distx
        elif self.method == 'glove':
            for j in range(self.nDocs):
                dv_2 = self.data['glove-vector'][j]
                if j!=ptIndex:
                    distx = self.getDistance(ptIndex,j)
                    distance[j] = distx
        elif self.method == 'transformer':
            for j in range(self.nDocs):
                dv_2 = self.data['transformer_vector'][j]
                if j!=ptIndex:
                    distx = self.getDistance(ptIndex,j)
                    distance[j] = distx
        
        # keeping only elements at a distnce of less than epsilon (or more for the cosine metric)
        if self.metric == 'cosine':
            tempDistances = {key: value for (key, value) in distance.items() if value > self.epsilon}
        else:
            tempDistances = {key:value for (key,value) in distance.items() if value < self.epsilon}

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

                neighbors = self.findNeighborOf(k)

                if neighbors:
                    self.clusterCount+=1
                    clusterName = f'cluster_{self.clusterCount}'
                    self.clusterMetadata[k] = clusterName
                    
                    # neighboring points of original point
                    for nbPoint in neighbors:
                        if self.clusterMetadata[nbPoint] == 'no_cluster':
                            self.clusterMetadata[nbPoint] = clusterName

                    innerNeighbors = self.findNeighborOf(k)

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

            
    def getDistance(self,docId_1,docId_2):

        if self.method == 'glove':
            dv_1 = self.data['glove-vector'][int(docId_1)]
            dv_2 = self.data['glove-vector'][int(docId_2)]
        elif self.method == 'tfidf':
            dv_1 = np.array(self.data['tfidf2vec-tfidf'][int(docId_1)])
            dv_2 = np.array(self.data['tfidf2vec-tfidf'][int(docId_2)])
        elif self.method == 'transformer':
            dv_1 = self.data['transformer_vector'][int(docId_1)]
            dv_2 = self.data['transformer_vector'][int(docId_2)]
            if self.metric == 'cosine':
                return np.dot(dv_1, dv_2) / (np.linalg.norm(dv_1) * np.linalg.norm(dv_2))

        return np.linalg.norm(dv_1-dv_2)

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
