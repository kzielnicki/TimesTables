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
            (lines,avgLength,stdLength,newlineRatio,rhymeQuotient,specialChar) = self.parameters
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
            print 'Possible poem w/ lines=%d, std=%f, nl_ratio=%f rhyming=%f num=%d\n\n' % self.parameters
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
    
    degree = 3 # polynomial degree for feature maping
    
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
                    print 'Loaded model trained on %d comments!' % self.m
            except Exception as e:
                print 'Exception: '+str(e)
                print 'Couldn\'t restore from file! Try training with:'
                print 'LearningModel(trainingSet)'
        else:
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
                
            self.X = self.featureMap(self.X)
                    
            # normalize the features in X
            self.mean = numpy.mean(self.X,0)
            self.stdev = numpy.std(self.X,0)
            self.X = self.normalize(self.X) #(self.X-self.mean)/self.stdev
            
            # randomize the order of X & y
            shuffleIdx = numpy.arange(self.m)
            numpy.random.shuffle(shuffleIdx)
            X_rand = self.X[shuffleIdx]
            y_rand = self.y[shuffleIdx]
            
            # then split into training, cross-validation, and test sets, with 60/20/20 split
            div1 = numpy.floor(self.m*0.6)
            div2 = numpy.floor(self.m*0.8)
            self.X_train = X_rand[0:div1]
            self.y_train = y_rand[0:div1]
            self.X_cv = X_rand[div1:div2]
            self.y_cv = y_rand[div1:div2]
            self.X_test = X_rand[div2:]
            self.y_test = y_rand[div2:]
            
            #self.svdVisualize()
            
            self.logit = LogisticRegression(C=1, penalty='l1')
            self.logit.fit(self.X_train,self.y_train)
            
            with open('model','w') as myFile:
                pickle.dump({'logit':self.logit,'mean':self.mean,'stdev':self.stdev,'n':self.n,'m':self.m},myFile)
    
    def featureMap(self,X):
        """
        Create higher order polynomial features up to 'degree'
        """
        
        
        multiply = lambda x,y: x*y # helper function for reduce
        idx = range(self.n)
        for d in range(2,self.degree+1):
            order = list(itertools.combinations_with_replacement(idx,d))
            for combination in order:
                newFeature = reduce(multiply,X[:,combination].T)
                # different behavior if newFeature is scalar or vector -- slightly ugly =(
                if isinstance(newFeature,numpy.ndarray):
                    X = numpy.c_[X,newFeature]
                else:
                    X = numpy.append(X,newFeature)
              
        
        return X
        
    def normalize(self,X):
        """
        return vector or point X normalized to zero mean and unit standard deviation
        """
        return (X-self.mean)/self.stdev
        
    def measureTestSet(self):
        pred_y = self.logit.predict(self.X_test)
        pred_prob = self.logit.predict_proba(self.X_test)[:,1]
        
        print 'F1 score = %f' % metrics.f1_score(self.y_test, pred_y)
        
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_test, pred_prob)
        # make thresholds start at 0
        thresholds = numpy.append([0],thresholds)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        plt.plot(thresholds, precision, 'b', thresholds, recall, 'r', thresholds, f1, 'g')
        plt.show()
    
    def reviewTrainingData(self, recheck=False):    
        """
        Calculate precision and recall on the training set, also display miscategorized poems,
        and optionally ask the user to re-confirm the categorization
        
        TODO: split into training data and test set for model evaluation
        """
        pred_y = self.logit.predict(self.X)
        pred_prob = self.logit.predict_proba(self.X)[:,1]
        
        # find and print the most significant coefficients from the logistic regression
        idx = range(self.n)
        order = []
        names = ('lines','ave','stdev','nlRatio','rhymeQ','sChar')
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
        
        idx = numpy.argsort(pred_prob)
        for n in range(len(idx)):
            i = idx[n]
            self.trainingSet[i].predProb = pred_prob[i]
            if pred_prob[i] < 0.1:
                if self.y[i]:
                    print '\n\n-------\n\n[False -] Miscategorized poem (p = %f):\n%s\n%s\n' % (pred_prob[i],self.trainingSet[i].parameters,self.trainingSet[i].url)
                    if recheck:
                        self.trainingSet[i].askIfPoem()
                    else:
                        print self.trainingSet[i].comment
            elif pred_prob[i] > 0.1:
                if not self.y[i]:
                    print '\n\n-------\n\n[False +] Miscategorized non-poem (p = %f):\n%s\n%s\n' % (pred_prob[i],self.trainingSet[i].parameters,self.trainingSet[i].url)
                    if recheck:
                        self.trainingSet[i].askIfPoem()
                    else:
                        print self.trainingSet[i].comment
        if recheck:     
            with open('trainingset','w') as myFile:
                pickle.dump(self.trainingSet,myFile)
    
    def svdVisualize(self):
        """
        Use SVD for feature visualization in 2D
        """
        
        (U,S,V) = numpy.linalg.svd(numpy.dot(self.X.T,self.X)/self.m)
        Z = numpy.zeros((self.m,2))
        Z[:,0] = numpy.dot(self.X,U[:,0])
        Z[:,1] = numpy.dot(self.X,U[:,1])
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
        

     
if __name__ == "__main__":
    """
    If run from command line, optionally update the training set,
    and retrain the learning model, based on parameters below:
    
    *** BEGIN PARAMETER DEFINITION
    """
    restore = True # true to restore comments from file, false to query API
    trainNewExamples = False # true to ask user to classify new examples, false to only use saved training set
    date = '20140107' # date to use comments from if training new examples (YYYYMMDD)
    """ END PARAMETER DEFINITION """
    
    
    # load training data
    try:
        with open('trainingset','r') as myFile:
            trainingSet = pickle.load(myFile)
    except:
        print 'Creating new training set'
        trainingSet = []
        
    # if requested, add new examples from 'date' to the training set
    if trainNewExamples:
        if len(trainingSet) > 100:
            myModel = LearningModel(trainingSet)
        else:
            myModel = None
            
        commentList = map(lambda x: x.comment, trainingSet)
        myComments = TimesComments(date,restore)
        
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