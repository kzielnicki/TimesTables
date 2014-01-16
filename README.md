TimesTables
===========

Reader comment analysis using the New York Times (c) community API

Installation
------------
* Requires python 2.7, numpy, matplotlib, scikit-learn and xmltodict
* Rename <tt>config.example</tt> to <tt>config.py</tt> and edit in your API keys
* Load <tt>from apiQuery import TimesComments</tt>, then <tt>TimesComments('20140105',False)</tt> to query the NYT API for comments from January 5th, 2014 (or choose another date)
* Load <tt>from classifyData import *</tt>, then <tt>classifyData('20140105')</tt> to begin classifying comments and training the learning model
* With learning model trained, <tt>from commentAnalysis import CommentAnalysis</tt>, then <tt>CommentAnalysis('20140105')</tt> to find poems and calculate word frequencies

Usage
-----
Use <tt>apiQuery.py</tt> to load new comments from the NYT API

Use <tt>classifyData.py</tt> to train the learning algorithm

Use <tt>commentAnalysis.py</tt> to perform analysis on comments

More Information
----------------
There is a [blog post](http://commentscount.blogspot.com/2014/01/sometimes-when-reading-times-comments.html) with more details about the ideas behind this project