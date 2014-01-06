from __future__ import division
from apiQuery import TimesComments
import pickle
import matplotlib.pyplot as plt
import numpy

class LabeledData:
    """
    Holds a comment and parameters describing the comment
    Tries to classify as a poem/not poem, asks user if unsure
    """
    
    def __init__(self, c, p):
        self.comment = c
        self.parameters = p
        self.isPoem = self.__checkIfPoem()
    
    # some things we'll just guess aren't poems, others we'll ask the user about
    def __checkIfPoem(self):
        (lines,stdLength,newlineRatio,rhymeQuotient,numerics) = self.parameters
        if lines < 2:
            return False
        if (newlineRatio+rhymeQuotient/100) < 0.02:
            #print 'few lines %f, few rhymes %f' % (newlineRatio, rhymeQuotient)
            return False
        if newlineRatio < 0.01:
            #print 'few lines %f < 0.1' % newlineRatio
            return False
    
        return self.__askIfPoem()
        
    def __askIfPoem(self):
        print '\n\n\n-------\n'
        print 'Possible poem w/ lines=%d, std=%f, nl_ratio=%f rhyming=%f num=%d\n\n' % self.parameters
        print self.comment+'\n'
        ans = ''
        while ans != 'y' and ans != 'n':
            ans = raw_input('Is this a poem (y/n)? ')
        if ans == 'y':
            return True
        else:
            return False
        
class LearningModel:
    """
    Create a logistic regression model to classify comments as poems or not poems
    """

    def __init__(self, trainingSet):
        """
        initialize from training set of labeledData, creating feature vector X and classification y
        """
        
        self.trainingSet = trainingSet
        self.m = len(trainingSet)
        self.X = numpy.zeros((self.m,5))
        self.y = numpy.zeros((self.m,1))
        
        i = 0
        for comment in trainingSet:
            self.X[i] = comment.parameters
            self.y[i] = comment.isPoem
            i += 1
            
        # normalize the features in X
        self.mean = numpy.mean(self.X,0)
        self.stdev = numpy.std(self.X,0)
        self.X = (self.X-self.mean)/self.stdev
        
        #self.svdVisualize()
        
        # apply unit bias
        #self.X = numpy.hstack((numpy.ones((self.m,1)), self.X))
        
    
    def svdVisualize(self):
        """
        use SVD for feature visualization in 2D
        """
        
        (U,S,V) = numpy.linalg.svd(numpy.dot(self.X.T,self.X)/self.m)
        Z = numpy.zeros((self.m,2))
        Z[:,0] = numpy.dot(self.X,U[:,0])
        Z[:,1] = numpy.dot(self.X,U[:,1])
        # plot projected data for visualization
        colors = map(lambda x: 'r' if x else 'b', self.y)
        plt.scatter(Z[:,0],Z[:,1],20,colors)
        plt.show()
        
        

        
if __name__ == "__main__":
    restore = True
    newExamples = False
    date = '20140104'
    
    if(restore):
        with open('timesComments'+date, 'r') as myFile:
            myComments = pickle.load(myFile)
        #myComments.analyzeComments()
    else:
        myComments = TimesComments()
        #myComments.initByKeyword('brainlike','computers learning')
        myComments.initByDate(date)
        with open('timesComments'+date,'w') as myFile:
            pickle.dump(myComments,myFile)
        #myComments.analyzeComments()
    
    # load training data and add new training data
    try:
        with open('trainingset','r') as myFile:
            trainingSet = pickle.load(myFile)
    except:
        print 'Creating new training set'
        trainingSet = []
    
    if newExamples:
        commentList = map(lambda x: x.comment, trainingSet)
        
        i = 0
        for comment in myComments.iterComments():
            i += 1
            if comment[0] in commentList:
                print 'comment already found!'
            else:
                print str(i) + '/' + str(len(myComments.myComments))
                trainingSet.append(LabeledData(comment[0],comment[1]))
        
        with open('trainingset','w') as myFile:
            pickle.dump(trainingSet,myFile)
    
    myModel = LearningModel(trainingSet)
    X = myModel.X
    y = myModel.y