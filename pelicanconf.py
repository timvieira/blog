#!/usr/bin/env python
# -*- coding: utf-8 -*- #
import os

AUTHOR = u'Tim Vieira'

SITENAME = u'Graduate Descent'
SITESUBTITLE = u''
SITEURL = 'http://timvieira.github.io/blog'  # change in publishconf.py

# Times and dates
DEFAULT_DATE_FORMAT = '%b %d, %Y'
TIMEZONE = 'US/Eastern'
DEFAULT_LANG = u'en'

# Set the article URL
ARTICLE_URL = 'post/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'post/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'

# Title menu options
MENUITEMS = [
    ('About', 'http://timvieira.github.io/'),
    ('Archive', '/blog/archives.html'),
]

NEWEST_FIRST_ARCHIVES = True

#Github include settings
GITHUB_USER = 'timvieira'
GITHUB_REPO_COUNT = 4
GITHUB_SKIP_FORK = True
GITHUB_SHOW_USER_LINK = True

# Blogroll
#LINKS =  (('Pelican', 'http://docs.notmyidea.org/alexis/pelican/'),
#          ('Python.org', 'http://python.org'),
#          ('Jinja2', 'http://jinja.pocoo.org'),
#          ('You can modify those links in your config file', '#'),)

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = 3

# STATIC_OUT_DIR requires https://github.com/jakevdp/pelican/tree/specify-static
#STATIC_OUT_DIR = ''
#STATIC_PATHS = ['images', 'figures', 'downloads']
#FILES_TO_COPY = [('favicon.png', 'favicon.png')]

# This requires Pelican 3.3+
STATIC_PATHS = ['images', 'figures', 'downloads', 'favicon.png']

CODE_DIR = 'code'
NOTEBOOK_DIR = 'notebook'

# Theme and plugins
#  Theme requires http://github.com/duilio/pelican-octopress-theme/
#  Plugins require http://github.com/getpelican/pelican-plugins/
THEME = './theme/pelican-octopress-theme'
PLUGIN_PATHS = ['./plugins/pelican-plugins']

# When creating a short summary of an article, this will be the default length
# in words of the text created. This only applies if your content does not
# otherwise specify a summary. Setting to None will cause the summary to be a
# copy of the original content.
SUMMARY_MAX_LENGTH = None

PLUGINS = [#'summary',
           'render_math',
           'liquid_tags.img',
           'liquid_tags.video',
           'liquid_tags.include_code',
           'liquid_tags.notebook',
           'liquid_tags.literal']


# The theme file should be updated so that the base header contains the line:
#
#  {% if EXTRA_HEADER %}
#    {{ EXTRA_HEADER }}
#  {% endif %}
#
# This header file is automatically generated by the notebook plugin
if not os.path.exists('_nb_header.html'):
    import warnings
    warnings.warn("_nb_header.html not found.  "
                  "Rerun make html to finalize build.")
else:
    EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')

# Sharing
TWITTER_USER = 'xtimv'
#GOOGLE_PLUS_USER = ''
#GOOGLE_PLUS_ONE = True
#GOOGLE_PLUS_HIDDEN = False
#FACEBOOK_LIKE = False
TWITTER_TWEET_BUTTON = True
TWITTER_LATEST_TWEETS = True
TWITTER_FOLLOW_BUTTON = True
TWITTER_TWEET_COUNT = 3
TWITTER_WIDGET_ID = '551816176788328448'
#TWITTER_SHOW_REPLIES = 'false'
#TWITTER_SHOW_FOLLOWER_COUNT = False

DISQUS_SITENAME = 'graduatedescent'

# RSS/Atom feeds
FEED_DOMAIN = '/blog' #SITEURL
FEED_ATOM = 'atom.xml'

# Search
SEARCH_BOX = False #True

#RELATIVE_URLS = False      #<<< needs to be True for disqus to work

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None

MATH_JAX = {
    'align': 'center',
    'macros': [
        #'/home/user/latex-macros.tex'
    ],
    #'source': "'//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'",
    #'source': "file:////home/timv/projects/blog/output/MathJax.js",
}
