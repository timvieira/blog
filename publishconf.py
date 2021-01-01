#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

# [2021-01-01 Fri] Disqus plugin wants absolute URLs; we could patch that.
SITEURL = 'https://timvieira.github.io/blog'
RELATIVE_URLS = False
#RELATIVE_URLS = True

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/category/%s.atom.xml'
TAG_FEED_ATOM = 'feeds/tag/%s.atom.xml'

#DELETE_OUTPUT_DIRECTORY = True

# Following items are often useful when publishing

DISQUS_SITEURL = 'http://timvieira.github.io/blog'
DISQUS_SITENAME = 'graduatedescent'

#GOOGLE_ANALYTICS = 'google-analytics-id'
