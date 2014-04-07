from __future__ import division
import pickle
import matplotlib.pyplot as plt
import numpy
import itertools
import pprint
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from apiQuery import TimesComments

class LabeledData:
    """
    Holds a comment and parameters describing the comment
    Tries to classify as a poem/not poem, asks user if unsure
    
    NOTE: requires potential user input on init!
    """
    
    def __init__(self, c, url, p, model=None, isPoem=None):
        self.manuallyClassified = False
        self.url = url
        self.comment = c
        self.parameters = p
        self.isPoem = isPoem
        self.predProb = None
        if model != None:
            self.predProb = model.predictNewPoem(p)
            
        if isPoem==None:
            self.checkIfPoem()
    
    def __str__(self):
        return '%s\n------\n%s' % (self.parameters,self.comment)

    def checkIfPoem(self):
        """
        If we have a prediction model available, use it to choose possible poems
        otherwise, make a rough guess based on line count, newlines, and rhyming
        """
        if self.predProb != None:
            if self.predProb > 0.01:
                self.askIfPoem()
            else:
                self.isPoem = False
        else:
            (lines,avgLength,stdLength,rhymeQuotient,numeric,specialChar) = self.parameters
            newlineRatio = 1/avgLength
            if lines < 2:
                self.isPoem = False
            elif (newlineRatio+rhymeQuotient/100) < 0.02:
                #print 'few lines %f, few rhymes %f' % (newlineRatio, rhymeQuotient)
                self.isPoem = False
            elif newlineRatio < 0.01:
                #print 'few lines %f < 0.1' % newlineRatio
                self.isPoem = False
            else:
                self.askIfPoem()
        
    def askIfPoem(self):
        """
        Ask the user if this comment is a poem, then label correctly
        """
        self.manuallyClassified = True
        
        print '\n-------\n'
        if self.predProb == None:
            print 'Possible poem w/ lines=%d, avglines=%f, std=%f, rhyming=%f, num=%d, special=%f \n\n' % self.parameters
        else:
            print 'Possible poem w/ probability=%f\n\n' % self.predProb
            
        print self.comment+'\n'
        ans = ''
        while ans != 'y' and ans != 'n':
            ans = raw_input('Is this a poem (y/n)? ')
        if ans == 'y':
            self.isPoem = True
        else:
            self.isPoem = False
        print '\n'
        
class LearningModel:
    """
    Create a logistic regression model to classify comments as poems or not poems
    """
    
    def __init__(self, trainingSet=None):
        """
        initialize from saved learning model, or
        initialize from training set of labeledData, creating feature vector X and classification y
        """
        
        if trainingSet == None:
            print 'Loading pre-trained model from file...'
            try:
                with open('model','r') as myFile:
                    modelDict = pickle.load(myFile)
                    self.n = modelDict['n']
                    self.m = modelDict['m']
                    self.mean = modelDict['mean']
                    self.stdev = modelDict['stdev']
                    self.logit = modelDict['logit']
                    self.degree = modelDict['degree']
                    self.useInverse = modelDict['useInverse']
                    print 'Loaded model trained on %d comments!' % self.m
            except Exception as e:
                print 'Exception: '+str(e)
                print 'Couldn\'t restore from file! Try training with:'
                print 'LearningModel(trainingSet)'
        else:
            self.degree = 2 # polynomial degree for feature mapping
            self.useInverse = True # whether to use inverse features for feature mapping
            self.C = 2 # regularization parameter 
            self.trainingSet = trainingSet
            self.n = len(trainingSet[0].parameters) # number of features
            self.m = len(trainingSet)
            self.X = numpy.zeros((self.m,self.n))
            self.y = numpy.zeros((self.m))
            
            print 'Training model on %d comments...' % self.m
            
            i = 0
            for comment in trainingSet:
                self.X[i] = comment.parameters
                self.y[i] = comment.isPoem
                i += 1
                
            
            # randomize the order of X & y
            self.shuffleIdx = numpy.arange(self.m)
            numpy.random.shuffle(self.shuffleIdx)
            self.y = self.y[self.shuffleIdx]
            
            self.mapAndNormalizeFeatures()
            
            #self.svdVisualize()
            
            self.logit = LogisticRegression(C=self.C, penalty='l1')
            self.logit.fit(self.X_train,self.y_train)
            
            self.pickleModel()
    
    def pickleModel(self):
        """
        Save a model trained on all data to a pickled file, with relevant parameters for predicting comments
        """
        print 'Saving model to file...'
        logit = LogisticRegression(C=self.C, penalty='l1')
        logit.fit(self.X_mapped,self.y)
            
        with open('model','w') as myFile:
            pickle.dump({'logit':logit,'degree':self.degree,'useInverse':self.useInverse,'mean':self.mean,'stdev':self.stdev,'n':self.n,'m':self.m},myFile)



    def mapAndNormalizeFeatures(self):
        """
        Prepare the dataset by mapping the features, splitting into training/cv/test,
        and normalizing mean & stdev
        """
        self.X_mapped = self.featureMap(self.X[self.shuffleIdx])
        
        # define splits for training, cross-validation, and test sets, with 60/20/20 split
        div1 = numpy.floor(self.m*0.6)
        div2 = numpy.floor(self.m*0.8)
                    
        # normalize the features in the training set
        self.mean = numpy.mean(self.X_mapped[0:div1],0)
        self.stdev = numpy.std(self.X_mapped[0:div1],0)
        self.X_mapped = self.normalize(self.X_mapped) #(self.X-self.mean)/self.stdev
        
        self.X_train = self.X_mapped[0:div1]
        self.y_train = self.y[0:div1]
        self.X_cv = self.X_mapped[div1:div2]
        self.y_cv = self.y[div1:div2]
        self.X_test = self.X_mapped[div2:]
        self.y_test = self.y[div2:]
    
    def featureMap(self,X):
        """
        Create higher order polynomial features up to 'degree'
        """
        # optionally, penalize inverse and higher order features more than 1st order features
        penaltyCoeff = 1.0
        penalties = penaltyCoeff*numpy.ones((self.n))
        
        # also define inverse features
        if self.useInverse:
            X = numpy.hstack((X,1/(X+0.00001)))
            penalties = numpy.append(penalties, penaltyCoeff*penalties);
        
        multiply = lambda x,y: x*y # helper function for reduce
        idx = range(2*self.n if self.useInverse else self.n)
        for d in range(2,self.degree+1):
            order = list(itertools.combinations_with_replacement(idx,d))
            for combination in order:
                newFeature = reduce(multiply,X[:,combination].T)
                # different behavior if newFeature is scalar or vector -- slightly ugly =(
                penalties = numpy.append(penalties, reduce(multiply,penalties[:,combination]))
                if isinstance(newFeature,numpy.ndarray):
                    X = numpy.c_[X,newFeature]
                else:
                    X = numpy.append(X,newFeature)
              
        self.penalties = penalties/penaltyCoeff
        return X
        
    def normalize(self,X):
        """
        Return vector or point X normalized to zero mean and unit standard deviation
        """
        return (X-self.mean)/self.penalties/self.stdev
    
    def cvTest(self):
        """
        Use cross validation set to choose parameters for fit
        """
        best_degree = None
        best_f1 = 0
        best_logit = None
        
        print '\n\nTest on degree = 1\n\n'
        self.degree = 1
        self.mapAndNormalizeFeatures()
        reg_params = 0.01*3**numpy.arange(0,13)
        for reg_param in reg_params:
            logit = LogisticRegression(C=reg_param, penalty='l1')
            logit.fit(self.X_train,self.y_train)
            
            pred_y = logit.predict(self.X_cv)
            f1 = metrics.f1_score(self.y_cv, pred_y)
            if f1>best_f1:
                best_c = reg_param
                best_f1 = f1
                best_logit = logit
                best_degree = 1
            
            pred_y = logit.predict(self.X_test)
            f1_test = metrics.f1_score(self.y_test, pred_y)
            
            print 'F1 score for c=%f:\t%f\t(testset = %f)' % (reg_param, f1,f1_test)
            
        
        
        
        print '\n\nTest on degree = 2\n\n'
        self.degree = 2
        self.mapAndNormalizeFeatures()
        reg_params = 0.01*3**numpy.arange(0,9)
        for reg_param in reg_params:
            logit = LogisticRegression(C=reg_param, penalty='l1')
            logit.fit(self.X_train,self.y_train)
            
            pred_y = logit.predict(self.X_cv)
            f1 = metrics.f1_score(self.y_cv, pred_y)
            if f1>best_f1:
                best_c = reg_param
                best_f1 = f1
                best_logit = logit
                best_degree = 2
            
            pred_y = logit.predict(self.X_test)
            f1_test = metrics.f1_score(self.y_test, pred_y)
            
            print 'F1 score for c=%f:\t%f\t(testset = %f)' % (reg_param, f1,f1_test)
            
            
            
        print '\n\nTest on degree = 3\n\n'    
        self.degree = 3
        self.mapAndNormalizeFeatures()  
        reg_params = 0.01*3**numpy.arange(0,8)
        for reg_param in reg_params:
            logit = LogisticRegression(C=reg_param, penalty='l1')
            logit.fit(self.X_train,self.y_train)
            
            pred_y = logit.predict(self.X_cv)
            f1 = metrics.f1_score(self.y_cv, pred_y)
            if f1>best_f1:
                best_c = reg_param
                best_f1 = f1
                best_logit = logit
                best_degree = 3
            
            pred_y = logit.predict(self.X_test)
            f1_test = metrics.f1_score(self.y_test, pred_y)
            
            print 'F1 score for c=%f:\t%f\t(testset = %f)' % (reg_param, f1,f1_test)
            
        print 'Best F1 score from cross-validation is %f, found on degree=%d, C=%f' % (best_f1, best_degree, best_c)
        self.degree = best_degree
        self.C = best_c
        self.logit = best_logit
        self.mapAndNormalizeFeatures()
        
        self.pickleModel()
    
    def measureTestSet(self, includeCV = False):
        if includeCV:
            X = numpy.r_[self.X_cv,self.X_test]
            y = numpy.append(self.y_cv,self.y_test)
        else:
            X = self.X_test
            y = self.y_test
            
        pred_y = self.logit.predict(X)
        pred_prob = self.logit.predict_proba(X)[:,1]
        
        print 'F1 score = %f' % metrics.f1_score(y, pred_y)
        
        precision, recall, thresholds = metrics.precision_recall_curve(y, pred_prob)
        # make thresholds start at 0
        thresholds = numpy.append([0],thresholds)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        plt.plot(thresholds, precision, 'b', label="Precision")
        plt.plot(thresholds, recall, 'r', label="Recall")
        plt.plot(thresholds, f1, 'g', label="F1")
        plt.legend(loc='lower left')
        plt.xlabel('Threshold')
        plt.show()
        
        targetThresholds = [0.01, 0.02, 0.04, 0.1, 0.2, 0.5]
        idxThreshold = numpy.zeros(len(targetThresholds),dtype=numpy.int32)
        idx = 0
        current = 0
        for t in thresholds:
            if t >= targetThresholds[current]:
                idxThreshold[current] = idx
                current += 1
                if current == len(targetThresholds):
                    break
            idx += 1
            
        #print idxThreshold
        #print precision
        #print precision[numpy.array([1.,2.,3.])]
                
        falsePosRates = (1-precision[idxThreshold])*100
        falseNegRates = (1-recall[idxThreshold])*100
        
        for (targetThreshold, falseNegRate, falsePosRate) in zip(targetThresholds, falseNegRates, falsePosRates):
            print 'With threshold of %d%%, per 100 poems detected, expected to miss %d, and incorrectly identify %d non-poems' % (100*targetThreshold, falseNegRate, falsePosRate)
    
    def reviewTrainingData(self, recheck=False):    
        """
        Calculate precision and recall on the training set, also display miscategorized poems,
        and optionally ask the user to re-confirm the categorization
        """
        pred_y = self.logit.predict(self.X_mapped)
        pred_prob = self.logit.predict_proba(self.X_mapped)[:,1]
        
        # find and print the most significant coefficients from the logistic regression
        idx = range(2*self.n if self.useInverse else self.n) # factor of 2 to account for inverse features
        order = []
        names = ('lines','aveLen','stdev','rhymeQ','#\'s','sChar')
        if self.useInverse:
            names = names + ('INV_lines','INV_aveLen','INV_stdev','INV_rhymeQ','INV_#\'s','INV_sChar')
        
        for d in range(1,self.degree+1):
            for group in itertools.combinations_with_replacement(idx,d):
                order.append([names[n] for n in group])

        coeff = self.logit.coef_.ravel()                
        sortedCoeff = sorted(zip(coeff,order), key = lambda tup: -abs(tup[0]))
        
        print 'Most significant coefficients:'
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(sortedCoeff)
        
        print 'F1 score = %f' % metrics.f1_score(self.y, pred_y)
        
        precision, recall, thresholds = metrics.precision_recall_curve(self.y, pred_prob)
        # make thresholds start at 0
        thresholds = numpy.append([0],thresholds)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        plt.plot(thresholds, precision, 'b', thresholds, recall, 'r', thresholds, f1, 'g')
        plt.show()
        
        shuffledTrain = [self.trainingSet[x] for x in self.shuffleIdx] # X and y have been shuffled, so the training set must be too
        idx = numpy.argsort(pred_prob)
        for n in range(len(idx)):
            i = idx[n]
            shuffledTrain[i].predProb = pred_prob[i]
            if pred_prob[i] < 0.2:
                if self.y[i]:
                    print '\n\n-------\n\n[False -] Miscategorized poem (p = %f):\n%s\n%s\n' % (pred_prob[i],shuffledTrain[i].parameters,shuffledTrain[i].url)
                    if recheck:
                        shuffledTrain[i].askIfPoem()
                    else:
                        print shuffledTrain[i].comment
            elif pred_prob[i] > 0.2:
                if not self.y[i]:
                    print '\n\n-------\n\n[False +] Miscategorized non-poem (p = %f):\n%s\n%s\n' % (pred_prob[i],shuffledTrain[i].parameters,shuffledTrain[i].url)
                    if recheck:
                        shuffledTrain[i].askIfPoem()
                    else:
                        print shuffledTrain[i].comment
        if recheck:     
            with open('trainingset','w') as myFile:
                pickle.dump(self.trainingSet,myFile)
    
    def svdVisualize(self):
        """
        Use SVD for feature visualization in 2D
        """
        
        (U,S,V) = numpy.linalg.svd(numpy.dot(self.X_mapped.T,self.X_mapped)/self.m)
        Z = numpy.zeros((self.m,2))
        Z[:,0] = numpy.dot(self.X_mapped,U[:,0])
        Z[:,1] = numpy.dot(self.X_mapped,U[:,1])
        # plot projected data for visualization
        colors = map(lambda x: 'r' if x else 'b', self.y)
        plt.scatter(Z[:,0],Z[:,1],20,colors)
        plt.show()
        
    def predictNewPoem(self,x):
        """
        Return probability that comment with feature vector x is a poem
        """
        x = numpy.array(x)
        x = self.featureMap(x)
        x = self.normalize(x)
        return self.logit.predict_proba(x)[:,1][0]
        

     
def classifyData(date=None):
    """
    Retrain the learning model and save to file, optionally also categorize new comments from 'date' with user input
    Returns trained model
    """
    
    # load training data
    try:
        with open('trainingset','r') as myFile:
            trainingSet = pickle.load(myFile)
    except:
        print 'Creating new training set'
        trainingSet = []
        
    # if requested, add new examples from 'date' to the training set
    if date != None:
        if len(trainingSet) > 100:
            myModel = LearningModel(trainingSet)
        else:
            myModel = None
            
        commentList = map(lambda x: x.comment, trainingSet)
        myComments = TimesComments(date)
        
        i = 0
        for comment in myComments.iterComments():
            i += 1
            if comment[0] in commentList:
                print 'comment already found!'
            else:
                print str(i) + '/' + str(len(myComments.myComments))
                newpt = LabeledData(comment[0],comment[1],comment[2],myModel)
                # if we have a trained learning model, only add manually classified points
                if newpt.predProb == None or newpt.manuallyClassified:
                    trainingSet.append(newpt)
        
        with open('trainingset','w') as myFile:
            pickle.dump(trainingSet,myFile)
    
    myModel = LearningModel(trainingSet)
    return myModel
    
    
if __name__ == "__main__":
    print 'Usage:'
    print '  classifyData() - retrain the model on the training set'
    print '  classifyData(\'YYYYMMDD\') - load and classify saved comments from date'
    print '  classifyData(\'YYYYMMDD\',False) - query API for comments from date, then classify'