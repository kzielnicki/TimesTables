from __future__ import division
import sys
import urllib, urllib2
import json as simplejson
import Queue
import threading
import string,re
import operator
import pickle
import time
import xmltodict
import itertools
import os
import functools
import numpy
import ConfigParser

class TimesComments:
    """
    Query the NYT API to obtain a list of comments, and perform basic analysis on the comments obtained
    
    Public methods:
    queryArticles   - get article list
    queryComments   - get comment list
    initByDate      - get all comments from a given date
    initByKeywords  - get all comments containing a given keyword
    iterComments    - return an iterable list of comments and comment parameters
    features        - return tuple of features calculated from comment
    analyzeComments - [DEPRECATED] do basic comment analysis, finding poems, calculating word frequency
    """
    commentsAPI = 'http://api.nytimes.com/svc/community/v2/comments/'
    articleAPI = 'http://api.nytimes.com/svc/search/v2/'
        
    def __init__(self, date=None, restore=True):
        """
        initialize a new TimesComments object, grabbing all comments from date if specified (format YYYYMMDD)
        optionally, set whether to restore from saved file, and savePoems to text file
        """
        self.myComments = []
        
        config = ConfigParser.ConfigParser()
        config.read('config.cfg')
        
        self.commentKey = config.get('API_KEY','commentKey')
        self.articleKey = config.get('API_KEY','articleKey')
        
        if date != None:
            if restore:
                try:
                    with open('timesComments'+date, 'r') as myFile:
                        saved = pickle.load(myFile)
                        self.myComments = saved.myComments
                        print 'Loaded %d comments!' % len(self.myComments)
                except Exception as e:
                    print 'Exception: '+str(e)
                    print 'Couldn\'t restore from file! Try loading from the API with:'
                    print 'TimesComments(\''+date+'\',False)'
                    
            else:
                #self.initByKeyword('brainlike','computers learning')
                self.initByDate(date)
                with open('timesComments'+date,'w') as myFile:
                    pickle.dump(self,myFile)
        

    def queryArticles(self, queryDict):
        """
        Query NYT API for articles using 'queryDict' to create the query string
        return a list of article names and URLs
        """
        queryDict['api-key'] = self.articleKey
        queryDict['fl'] = 'headline,web_url'

        url = self.articleAPI + 'articlesearch.json?%(query)s' % {'query': urllib.urlencode(queryDict)}
        search_results = urllib.urlopen(url)

        json = simplejson.loads(search_results.read())
        assert json['status'] == 'OK'
        results = json['response']

        #print simplejson.dumps(json, indent=4)

        articleList = []
        
        for i in results['docs']:
            print i['headline']['main']
            articleList.append(i['web_url'])

        return articleList

        
    def __queryCommentsHelper(self, commentQ, errorQ, queryDict, offset, attempt=1):
        """
        Do the actual work of querying the comment API, 25 comments at a time, at position 'offset'
        
        designed for multi-threading, successfully obtained comments are put in 'commentQ'
        if comments are not obtained, 'offset' is put in 'errorQ' for later re-try
        
        (using XML because JSON deletes line breaks for some reason!)
        """
        if attempt > 5:
            print 'TOO MANY FAILED ATTEMPTS FOR OFFSET %d, ABANDONING!' %offset
            return 0
        
        print 'Querying API with offset = '+str(offset)
        queryDict['offset'] = offset
        
        #url is built differently based on whether we are searching by date
        url = self.commentsAPI
        if 'date' in queryDict:
            url += 'by-date/%(date)s.xml?' % queryDict
        else:
            url += 'url/exact-match.xml?'
        
        url += urllib.urlencode(queryDict)
        try:
            search_results = urllib2.urlopen(url, None, 5)
            json = xmltodict.parse(search_results.read())['result_set']
        except Exception as e:
            print 'Exception: '+str(e)
            print 'Couldn\'t get reply to query '+url
            errorQ.put((offset,attempt))
            #print search_results.read()
            #print simplejson.dumps(json, indent=4)
            return 0
            
        #print simplejson.dumps(json, indent=4)
        assert json['status'] == 'OK'
        results = json['results']
        
        # if len(results['comments']['comment']) < 25:
            # print 'NOTE: only %d comments for offset %d!' % (len(results['comments']['comment']), offset)
        
        #print simplejson.dumps(results['comments']['comment'], indent=4)
        try:
            for comment in results['comments']['comment']:
                #print comment['commentBody'] + "\n\n------\n\n"
                edited = comment['commentBody'].encode('utf-8')
                edited = re.sub('<br/>', '\n', edited)
                # edited = re.sub('<[^<]+?>', '', edited)
                commentQ.put({'comment':edited,'id':comment['commentSequence'],'url':comment['articleURL']})
            return int(results['totalCommentsFound'])
        except Exception as e:
            print 'Exception: '+str(e)
            print 'Query %s returned unexpected results' % url
            simplejson.dumps(results)
            errorQ.put((offset,attempt))
            return 0
            
    
    def queryComments(self, queryDict):
        """
        query the NYT API for comments with a query string determined by 'queryDict'
        """
        
        queryDict['api-key'] = self.commentKey
        queryDict['sort'] = 'oldest'
        commentQ = Queue.Queue()
        errorQ = Queue.Queue()
  
        totalFound = self.__queryCommentsHelper(commentQ, errorQ, queryDict, 0)
        print 'found '+str(totalFound)+' comments'

        # now keep querying (in separate threads) to get all of the comments
        count = 0
        for offset in range(25,totalFound,25):
            count += 1  # make sure we don't send queries too quickly
            if count > 25:
                count = 0
                time.sleep(3)
            t = threading.Thread(target=self.__queryCommentsHelper, args = (commentQ, errorQ, queryDict, offset))
            t.daemon = True
            t.start()
            # self.__queryCommentsHelper(commentQ, queryDict, offset)
            
        # retry everything in the errorQ
        while not errorQ.empty():
            (offset,attempt) = errorQ.get()
            print 'Retrying after error #%d' % attempt
            self.__queryCommentsHelper(commentQ, errorQ, queryDict, offset,attempt+1)
            

        for i in range(totalFound):
            # try to get all of the comments, waiting up to 5 seconds
            try:
                self.myComments.append(commentQ.get(True,5))
            except Queue.Empty:
                print 'Couldn\'t get all the comments? Try running again.'
                break
            
        # make sure we got at least as many comments as we wanted (maybe a couple more people posted)
        print 'got %d comments (expected %d)' % (len(self.myComments), totalFound)
        if  len(self.myComments) < totalFound:
            ans = None
            while ans != 'y' and ans != 'n':
                ans = raw_input('Missed some comments, continue anyways (y/n)? ')
            if ans == 'n':
                sys.exit()


    def initByDate(self, date):
        """
        initialze the comment list, grabbing all comments from 'date'
        **Date format is YYYYMMDD**
        """
        self.myComments = []
        self.queryComments({'date':date})

    def initByKeyword(self, keyword, headlineFilter=''):
        """
        initialze the comment list, grabbing all comments matching 'keyword'
        and optionally in articles with headlines containing 'headlineFilter'
        """
        self.myComments = []
        query = {'q' : keyword}
        if headlineFilter != '':
            query['fq'] = 'headline:("'+headlineFilter+'")'
        resultURLs =  self.queryArticles(query)
        for url in resultURLs:
            query = {'url': url}
            self.queryComments(query)

    @staticmethod
    def __sharedLetters(word1,word2):
        """ 
        return the number of end letters shared between two words, ignoring case
        ex: __sharedLetters('XyAbC', 'aaAAabc') = 2
        if words are identical, returns 1
        """
        word1 = word1.lower()
        word2 = word2.lower()
        # return 1 if the two words are actually identical
        if word1 == word2:
            return 1
            #print word1 + ' == ' + word2
            
        # reverse the strings to put last letters first
        word1 = word1[::-1]
        word2 = word2[::-1]
        
        for i in range(min(len(word1),len(word2))):
            if word1[i] != word2[i]:
                return i

        return min(len(word1),len(word2))
        
    @staticmethod
    def __rhymeQuotient(comment):
        """
        Calculate the "rhymy-ness" of a comment, basically how many line ending words
        are similar to other line-ending words
        """
        #strip punctuation and numbers
        comment = comment.translate(string.maketrans("",""), string.punctuation)
        comment = comment.translate(string.maketrans("",""), string.digits)
        
        lines = comment.split('\n')
        
        lastWords = []
        for line in lines:
            words = line.split()
            if len(words) >= 1:
                lastWords.append(words[-1])
                
        # can't have rhymes if we have fewer than two lines
        if len(lastWords) < 2:
            return 0
        
        #print lastWords
        # now score each word by similarity with a following word
        for i in range(len(lastWords)):
            best = 0
            for j in range(i+1,len(lastWords)):
                best = max(best, TimesComments.__sharedLetters(lastWords[i],lastWords[j]))
            lastWords[i] = best
            
        lastWords = map(lambda x: 5 if x >= 2 else x, lastWords)
        return sum(lastWords)/(len(lastWords)-1)
    
    @staticmethod
    def features(comment):
        """
        Calculate the feature vector for a given comment
        """
        lineList = comment.split('\n')
        lineList = numpy.array(map(lambda x: len(x), lineList))
        
        lines = len(lineList)
        avgLength = numpy.mean(lineList)
        stdLength = numpy.std(lineList)
        #newlineRatio = lines/len(comment) # basically equal to 1/avgLength
        rhymeQ = TimesComments.__rhymeQuotient(comment)
        numeric = len(filter(functools.partial(operator.contains, string.digits), comment))
        specChar = len(filter(functools.partial(operator.contains, '@=&#$%^*<>/~\\+'), comment))
        
        return (lines,avgLength,stdLength,rhymeQ,numeric,specChar)
        
        
    def iterComments(self):
        """ 
        return an iterable list of comments along with calculated parameters
        format: (comment, (param1, param2...))
        """
        for commentProperties in self.myComments:
            comment = commentProperties['comment']
            url = '%s?comments#permid=%s' % (commentProperties['url'],commentProperties['id'])
            #print url
            #print comment
            #text = url + '\n\n' + comment.decode('utf-8').encode("ascii","ignore")
            
            yield (comment.decode('utf-8').encode("ascii","ignore"), url, self.features(comment))
            
        
        
    def analyzeComments(self):
        """
        Analyze comments for word frequency, also find and optionally save poems
        
        DEPRECATED! USE commentAnalysis.py INSTEAD!
        """
        wordBucket = {}
        
        for commentProperties in self.myComments:
            comment = commentProperties['comment']
            comment = re.sub('\n+', '\n', comment)
            newlineRatio = comment.count('\n')/len(comment)
            # if comment.find('humans to train') >= 0:
                # comment = re.sub('<br/>', '\n', comment)
                # print comment
                # print newlineRatio
                
            if newlineRatio > 0.02:
                print comment
                print '\nrhymeQuotient=%f, newlineRatio=%f' % (TimesComments.__rhymeQuotient(comment), newlineRatio)
                print '%s?comments#permid=%s' % (commentProperties['url'],commentProperties['id'])
                print '\n\n\n--------\n\n\n'
                    
        
            words = comment.split()
            for word in words:
                word = word.lower()
                new_word = word.translate(string.maketrans("",""), string.punctuation)
                if new_word in wordBucket:
                    wordBucket[new_word] += 1
                else:
                    wordBucket[new_word] = 1
                    
        sortedWords = sorted(wordBucket.iteritems(), key=operator.itemgetter(1), reverse = True)
        print sortedWords[:100]
        #return sortedWords
        


if __name__ == "__main__":
    print 'Usage:'
    print '  TimesComments(\'YYYYMMDD\') - restore comments from date'
    print '  TimesComments(\'YYYYMMDD\',False) - query API for comments from date, then save to file'
    #myComments = TimesComments('20140104')

