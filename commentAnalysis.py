from apiQuery import TimesComments
from classifyData import *
import string
import operator

class CommentAnalysis:
    """
    Provides an interface to apiQuery and classifyData to grab and analyze poems from a given date
    """
    
    def __init__(self,date,savePoemsToFile=False,restore=True):
        """
        Initialize comment analysis object by loading comments from 'date' and perform analysis
        Optionally choose whether to save found poems, and whether to restore saved comments from file
        NOTE: poetry finding assumes learning model has already been trained & saved
        """
        self.date = date
        self.myComments = TimesComments(date,restore)
        self.myModel = LearningModel()
        
        print 'Beginning comment analysis...\n\n\n'
        self.findPoems(savePoemsToFile)
        self.wordFrequency()
    
    def findPoems(self,saveToFile):
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
                    with open('poems'+self.date,'a') as f:
                        f.write(comment+'\n')
                        f.write('\nPossible poem w/ probability=%f\n' % predProb)
                        f.write('%s?comments#permid=%s\n' % (commentProperties['url'],commentProperties['id']))
                        f.write('\n\n\n--------\n\n\n\n')
    
    def wordFrequency(self):
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
        print sortedWords[:100]
        return sortedWords
        
        

if __name__ == "__main__":
    print 'Usage:'
    print '  CommentAnalysis(\'YYYYMMDD\') - analyze saved comments from date'
    print '  CommentAnalysis(\'YYYYMMDD\',True) - analyze saved comments from date, saving poems to file'
    print '  CommentAnalysis(\'YYYYMMDD\',False,False) - query API for comments from date, then analyze'
    #myAnalysis = AnalyzeComments('20140104')