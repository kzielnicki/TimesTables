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
            print 'Top words from today:\n\n'
            self.wordFrequency(saveToFile)
            print '\nComparing words with yesterday:\n\n'
            multi = MultiAnalysis(date,1,saveToFile,verbose)
            multi.gainsAndLosses(True)
    
    def findPoems(self,saveToFile=False):
        """
        Find and optionally save poems to file using learning model
        """
        self.myModel = LearningModel()

        if saveToFile:
            with open('poems'+self.date,'a') as f:
                f.write('Comment poems from %s\n\n' % self.date)

        foundPoems = [] 
        for commentProperties in self.myComments.myComments:
            comment = commentProperties['comment']
            #comment = re.sub('\n+', '\n', comment)
           
            predProb = self.myModel.predictNewPoem(TimesComments.features(comment))
                
            # display everything with 20%+ chance of being a poem
            if predProb > 0.2:
                foundPoems.append((predProb, comment, '%s?comments#permid=%s\n' % (commentProperties['url'],commentProperties['id'])))
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
                        
        print 'Found %d poems!\n\n' % len(foundPoems)
        return foundPoems
        
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
        
            if self.verbose:
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(sortedWords[:100])

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
    
    def __init__(self,date=None,lookbackDays=None,saveToFile=None,verbose=True,interactive=False):
        # we can optionally ask the user about parameters
        self.verbose = verbose

        if interactive:
            if date == None:
                date = raw_input('Analyze comments from what date (YYYYMMDD)? ')
            if lookbackDays == None:
                lookbackDays = int(raw_input('How many days to look back for comparison? '))
            if saveToFile == None:
                ans = ''
                while ans != 'y' and ans != 'n':
                    ans = raw_input('Save output to file (y/n)? ')
                if ans == 'y':
                    self.saveToFile = True
                else:
                    self.saveToFile = False
        elif saveToFile == None:
            self.saveToFile = False
        else:
            self.saveToFile = saveToFile

        dateObj = datetime.datetime.strptime(date,'%Y%m%d').date()
        day = datetime.timedelta(days=1)

        wordDict = {}
        self.analyzers = []
        self.words = []
        self.dates = []
        for i in range(lookbackDays+1):
            date_str = dateObj.strftime('%Y%m%d')
            self.dates.append(date_str)
            dateObj -= day
            print 'Loading ' + date_str
            analyzer = CommentAnalysis(date_str,verbose=False)
            self.analyzers.append(analyzer)
            words = analyzer.wordFrequency()
            self.words.append(words)
            #print words['the']
            # combine counts from all days
            wordDict = dict( (n, wordDict.get(n, 0)+words.get(n, 0)) for n in set(wordDict)|set(words) )
            #print wordDict['the']

            
        # sort by number of occurrences
        self.wordDict = wordDict
        self.sortedWords = sorted(wordDict.iteritems(), key=operator.itemgetter(1), reverse = True)
        self.totalWords = reduce(lambda (a,b),(c,d): ('sum',b+d),self.sortedWords)[1]

        # check whether to go ahead and do the analysis right now
        if interactive:
            ans = ''
            while ans != 'y' and ans != 'n':
                ans = raw_input('Plot word frequencies (y/n)? ')
            if ans == 'y':
                self.powerLawPlot()
                
            ans = ''
            while ans != 'y' and ans != 'n':
                ans = raw_input('Calculate gains and losses (y/n)? ')
            if ans == 'y':
                self.gainsAndLosses()


    @staticmethod
    def __H(n,m):
        """ Generalized Harmonic Number """
        series = (numpy.arange(n) + 1.0) ** -m
        return sum(series)

    @staticmethod
    def getWilsonInterval(f,N,z=2):
        """
        95% confidence interval for a frequency f assuming binomial confidence interval
        """
        # wilson interval is K*(A +/- B) where:
        K = 1/(1+z**2/N)
        A = f + z**2/(2*N)
        B = z*numpy.sqrt(f*(1-f)/N + z**2/(4*N**2))

        low = K*(A-B)
        high = K*(A+B)
        return (low,high)

    def gainsAndLosses(self, z=2):
        """
        Find words that have gained or lost significantly in frequency
        By default, compares with all days in analysis set, optionally only to yesterday
        z=2 for ~95% confidence interval
        """
        yesterday = None
        result = {}
        date = self.dates[0]
        words = self.words[0]

        change = {}
        total = reduce(lambda (a,b),(c,d): ('sum',b+d),words.iteritems())[1]

        for (word,count) in words.iteritems():
            freq = count/total
            (low_f, high_f) = MultiAnalysis.getWilsonInterval(freq,total,z)

            N = self.totalWords - total # don't include today's words
            basecount = self.wordDict[word]-count
            basefreq = max(basecount,1) / N # make sure basefreq isn't 0 to avoid infinities

            (base_low,base_high) = MultiAnalysis.getWilsonInterval(basefreq, N, z)

            change[word] = (freq / basefreq, low_f/base_high, high_f/base_low, count, basecount)
            
        self.change = change
        gainers = sorted(change.iteritems(), key=lambda (k,v): v[1], reverse = True)
        losers = sorted(change.iteritems(), key=lambda (k,v): v[2])

            
        output = ''
        output += 'Gainers:\n'
        for (word, vals) in gainers[:10]:
            output += '%s\t%f\t%f\t%f\t%i\t%i\n' % ((word,)+vals)
        output += '\nLosers:\n'
        for (word, vals) in losers[:10]:
            output += '%s\t%f\t%f\t%f\t%i\t%i\n' % ((word,)+vals)

        if self.verbose:
            print '\n\n------%s------\n\n' % date
            print output
        if self.saveToFile:
            with open('trending'+date,'w') as f:
                f.write(output)

        result = (gainers, losers)

        return result

    def powerLawPlot(self):
        counts = [count for (word,count) in self.sortedWords if count > 10]
        #total = sum(counts)
        wordFreq = numpy.array([count/self.totalWords for count in counts])
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
    print '  MultiAnalysis(interactive=True) - analyze word frequency trends from multiple days'
    #myAnalysis = AnalyzeComments('20140104')