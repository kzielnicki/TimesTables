from __future__ import division
from apiQuery import TimesComments
from classifyData import *
import string
import datetime
import operator
import pprint
import numpy
import matplotlib.pyplot as plt

class CommentAnalysis:
    """
    Provides an interface to apiQuery and classifyData to grab and analyze poems from a given date
    """
    
    def __init__(self,date,saveToFile=False,restore=True,verbose=True):
        """
        Initialize comment analysis object by loading comments from 'date' and perform analysis
        Optionally choose whether to save analysis, and whether to restore saved comments from file
        NOTE: poetry finding assumes learning model has already been trained & saved
        """
        self.date = date
        self.myComments = TimesComments(date,restore)
        
        self.verbose = verbose
        if verbose:
            print 'Beginning comment analysis...\n\n\n'
            self.findPoems(saveToFile)
            self.wordList = self.wordFrequency(saveToFile)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self.wordList[:100])
    
    def findPoems(self,saveToFile=False):
        """
        Find and optionally save poems to file using learning model
        """
        self.myModel = LearningModel()

        if saveToFile:
            with open('poems'+self.date,'a') as f:
                f.write('Comment poems from %s\n\n' % self.date)

        poemsFound = 0
        for commentProperties in self.myComments.myComments:
            comment = commentProperties['comment']
            #comment = re.sub('\n+', '\n', comment)
           
            predProb = self.myModel.predictNewPoem(TimesComments.features(comment))
                
            # display everything with 20%+ chance of being a poem
            if predProb > 0.2:
                poemsFound += 1
                if self.verbose:
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
                        
        print 'Found %d poems!\n\n' % poemsFound
        
    def wordFrequency(self,saveToFile=False):
        """
        Analyze comments for word frequency
        """
        
        wordDict = {}

        for commentProperties in self.myComments.myComments:
            comment = commentProperties['comment']
            #comment = re.sub('\n+', '\n', comment)
            
            words = comment.split()
            for word in words:
                word = word.lower()
                new_word = word.translate(string.maketrans("",""), string.punctuation)
                if new_word != '':
                    wordDict[new_word] = 1 + wordDict.get(new_word,0)
                        
        if saveToFile or self.verbose:
            # sort by number of occurrences
            sortedWords = sorted(wordDict.iteritems(), key=operator.itemgetter(1), reverse = True)
            total = reduce(lambda (a,b),(c,d): ('sum',b+d),sortedWords)[1] # total number of occurrences of all words in list

            # compute word frequencies
            sortedWords = [(word,count,count/total) for (word,count) in sortedWords]
        
            # if requested, save all words that appear at least 10 times, along with count and frequency
            if saveToFile:
                with open('wordcount'+self.date,'w') as f:
                    for (word,count,freq) in sortedWords:
                        if count >= 10:
                            f.write('%s, %d, %f\n' % (word,count,freq))
        
        return wordDict

class MultiAnalysis:
    """
    Perform analysis on comments from multiple days
    """
    
    def __init__(self,startDate,numDays):
        dateObj = datetime.datetime.strptime(startDate,'%Y%m%d').date()
        day = datetime.timedelta(days=1)

        wordDict = {}
        self.analyzers = []
        for i in range(numDays):
            date = dateObj.strftime('%Y%m%d')
            dateObj += day
            print 'Loading ' + date
            analyzer = CommentAnalysis(date,verbose=False)
            self.analyzers.append(analyzer)
            words = analyzer.wordFrequency()
            #print words['the']
            # combine counts from all days
            wordDict = dict( (n, wordDict.get(n, 0)+words.get(n, 0)) for n in set(wordDict)|set(words) )
            #print wordDict['the']

            
        # sort by number of occurrences
        self.sortedWords = sorted(wordDict.iteritems(), key=operator.itemgetter(1), reverse = True)

    @staticmethod
    def __H(n,m):
        """ Generalized Harmonic Number """
        series = (numpy.arange(n) + 1.0) ** -m
        return sum(series)

    @staticmethod
    def getZipfConfint(f,N):
        """
        95% confidence interval for frequency f drawn from zipf distribution with N samples
        """
        #zipf = ((numpy.arange(n)+1.0) ** -1) / MultiAnalysis.__H(n,1)

        # transform frequencies to half-normal distribution with sigma=1
        f_to_p = lambda f: numpy.sqrt( -2 * numpy.log(f) )
        p_to_f = lambda p: numpy.exp( -(p**2) / 2 )

        # variance for half-normal distribution is sigma*sqrt(1-2/Pi)
        p = f_to_p(f)
        stdErr = 2*numpy.sqrt(1-2/numpy.pi)/numpy.sqrt(N)
        print 'p=%f, stderr=%f' % (p,stdErr)
        
        low_p = p - stdErr
        high_p = p + stdErr
        print 'low %f, high %f' % (low_p,high_p)


        low_f = p_to_f(high_p)
        high_f = p_to_f(low_p)
        print 'low %f, high %f' % (low_f,high_f)

        return(low_f,high_f)


    def powerLawPlot(self):
        counts = [count for (word,count) in self.sortedWords if count > 10]
        total = sum(counts)
        wordFreq = numpy.array([count/total for count in counts])
        n = len(wordFreq)
        rank = numpy.arange(1,n+1)

        #fitfunc = lambda p, x: p[0]*cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function
        #errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

        H = MultiAnalysis.__H(n,1)
        # zipf's law for word frequencies is 1/(k*H) where k is word rank
        zipf = lambda k: 1/(k*H)

        plt.loglog(rank, wordFreq,'bx',label='Word Frequency in NYT Comments')
        plt.loglog(rank, zipf(rank),'r',label='Zipf\'s Law')
        plt.xlabel('Word Rank')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()

        #mu = MultiAnalysis.__H(n,2) / (H**2)
        #sigma = numpy.sqrt(MultiAnalysis.__H(n,3) / (H**3) - mu**2)



        

if __name__ == "__main__":
    print 'Usage:'
    print '  CommentAnalysis(\'YYYYMMDD\') - analyze saved comments from date'
    print '  CommentAnalysis(\'YYYYMMDD\',True) - analyze saved comments from date, saving poems to file'
    print '  CommentAnalysis(\'YYYYMMDD\',False,False) - query API for comments from date, then analyze'
    #myAnalysis = AnalyzeComments('20140104')