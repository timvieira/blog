#!/usr/bin/env python
# -*- coding: utf-8 -*- #

AUTHOR = u'Tim Vieira'
SITENAME = 'Graduate Descent'
SITEURL = 'http://timvieira.github.io/blog'

OUTPUT_PATH = 'output/blog/'

TIMEZONE = 'US/Eastern'
DEFAULT_LANG = u'en'
DEFAULT_DATE_FORMAT = '%b %d, %Y'

# RSS/Atom feeds
FEED_DOMAIN = '/blog' #SITEURL
FEED_ATOM = 'atom.xml'

PATH = 'content'

#THEME = '../theme/brutalist'
#THEME = '../theme/TuftePelican/'
#THEME = '../theme/fabianp-tufte-pelican/'
THEME = '../theme/pelican-octopress-theme'

# Set the article URL
ARTICLE_URL = 'post/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'post/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'

ARCHIVES_SAVE_AS = 'index.html'
INDEX_SAVE_AS = 'index2.html'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = []

# Social widget
SOCIAL = []

DEFAULT_PAGINATION = 1

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = False

# When creating a short summary of an article, this will be the default length
# in words of the text created. This only applies if your content does not
# otherwise specify a summary. Setting to None will cause the summary to be a
# copy of the original content.
SUMMARY_MAX_LENGTH = None

PLUGIN_PATHS = [
    '../plugins/pelican-plugins',
    '../plugins',
]

PLUGINS = ['render_math',
           'liquid_tags.img',
           #'liquid_tags.video',
           #'liquid_tags.include_code',
           'ipynb.liquid',
           #'liquid_tags.notebook',
           'liquid_tags.literal']

STATIC_PATHS = ['images', 'figures', 'downloads', 'favicon.png']

CODE_DIR = 'code'
NOTEBOOK_DIR = 'notebook'

# Title menu options
MENUITEMS = [
    ('About', 'http://timvieira.github.io/'),
    ('Archive', '/blog/index.html'),
]

NEWEST_FIRST_ARCHIVES = True

DISQUS_SITENAME = 'graduatedescent'

# <octopress>
DISPLAY_CATEGORIES_ON_MENU = False
# </octopress>


#Github include settings
#GITHUB_USER = 'timvieira'
#GITHUB_REPO_COUNT = 4
#GITHUB_SKIP_FORK = True
#GITHUB_SHOW_USER_LINK = True

# This requires Pelican 3.3+
#STATIC_PATHS = ['images', 'figures', 'downloads', 'favicon.png']

#GOOGLE_PLUS_USER = ''
#GOOGLE_PLUS_ONE = True
#GOOGLE_PLUS_HIDDEN = False
#FACEBOOK_LIKE = False

TWITTER_USER = 'xtimv'
#TWITTER_TWEET_BUTTON = True
TWITTER_LATEST_TWEETS = False
#TWITTER_TWEET_COUNT = 3
TWITTER_FOLLOW_BUTTON = True
#TWITTER_WIDGET_ID = '551816176788328448'
#TWITTER_SHOW_REPLIES = 'false'
TWITTER_SHOW_FOLLOWER_COUNT = True


#FEED_ATOM = False
