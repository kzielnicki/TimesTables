from __future__ import division
from apiQuery import TimesComments
from classifyData import *
import string
import operator
import pprint
import matplotlib.pyplot as plt

class CommentAnalysis:
    """
    Provides an interface to apiQuery and classifyData to grab and analyze poems from a given date
    """
    
    def __init__(self,date,saveToFile=False,restore=True):
        """
        Initialize comment analysis object by loading comments from 'date' and perform analysis
        Optionally choose whether to save analysis, and whether to restore saved comments from file
        NOTE: poetry finding assumes learning model has already been trained & saved
        """
        self.date = date
        self.myComments = TimesComments(date,restore)
        self.myModel = LearningModel()
        
        print 'Beginning comment analysis...\n\n\n'
        self.findPoems(saveToFile)
        self.wordList = self.wordFrequency(saveToFile)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.wordList[:100])
    
    def findPoems(self,saveToFile=False):
        """
        Find and optionally save poems to file using learning model
        """

        for commentProperties in self.myComments.myComments:
            comment = commentProperties['comment']
            #comment = re.sub('\n+', '\n', comment)
           
            predProb = self.myModel.predictNewPoem(TimesComments.features(comment))
                
            # display everything with 20%+ chance of being a poem
            if predProb > 0.2:
                print 'Possible poem w/ probability=%f\n\n' % predProb
                print comment
                print '\n%s?comments#permid=%s' % (commentProperties['url'],commentProperties['id'])
                print '\n\n\n--------\n\n\n'
                if saveToFile:
                    with open('poems'+self.date,'w') as f:
                        f.write(comment+'\n')
                        f.write('\nPossible poem w/ probability=%f\n' % predProb)
                        f.write('%s?comments#permid=%s\n' % (commentProperties['url'],commentProperties['id']))
                        f.write('\n\n\n--------\n\n\n\n')
    
    def wordFrequency(self,saveToFile=False):
        """
        Analyze comments for word frequency
        """
        
        wordBucket = {}

        for commentProperties in self.myComments.myComments:
            comment = commentProperties['comment']
            #comment = re.sub('\n+', '\n', comment)
            
            words = comment.split()
            for word in words:
                word = word.lower()
                new_word = word.translate(string.maketrans("",""), string.punctuation)
                if new_word != '':
                    if new_word in wordBucket:
                        wordBucket[new_word] += 1
                    else:
                        wordBucket[new_word] = 1
                        
        sortedWords = sorted(wordBucket.iteritems(), key=operator.itemgetter(1), reverse = True)
        total = reduce(lambda (a,b),(c,d): ('sum',b+d),sortedWords)[1]
        
        # if requested, save all words that appear at least 10 times, along with count and frequency
        if  True or saveToFile:
            with open('wordcount'+self.date,'w') as f:
                for (word,count) in sortedWords:
                    if count >= 10:
                        freq = count/total
                        f.write('%s, %d, %f\n' % (word,count,freq))
        
        return sortedWords
        
        

if __name__ == "__main__":
    print 'Usage:'
    print '  CommentAnalysis(\'YYYYMMDD\') - analyze saved comments from date'
    print '  CommentAnalysis(\'YYYYMMDD\',True) - analyze saved comments from date, saving poems to file'
    print '  CommentAnalysis(\'YYYYMMDD\',False,False) - query API for comments from date, then analyze'
    #myAnalysis = AnalyzeComments('20140104')