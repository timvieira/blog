PELICAN=pelican
PELICANOPTS=

BASEDIR=$(CURDIR)
INPUTDIR=$(BASEDIR)/content
OUTPUTDIR=$(BASEDIR)/output/blog
CONFFILE=$(BASEDIR)/pelicanconf.py
PUBLISHCONF=$(BASEDIR)/publishconf.py
DEPLOYREPOSITORY=timvieira.github.io

html: clean $(OUTPUTDIR)/index.html
	@echo 'Done'

$(OUTPUTDIR)/%.html:
	$(PELICAN) $(INPUTDIR) -o $(OUTPUTDIR) -s $(CONFFILE) $(PELICANOPTS)

clean: $(OUTPUTDIR)
	find $(OUTPUTDIR) -mindepth 1 -delete

push: html
	rm -rf ~/projects/self/timvieira.github.com/blog
	mkdir ~/projects/self/timvieira.github.com/blog
	rsync -a $(OUTPUTDIR)/. ~/projects/self/timvieira.github.com/blog/.
	( cd ~/projects/self/timvieira.github.com/ && hg addremove && hg ci -m 'update blog' && hg bookmarks -r tip master && hg push )

$(OUTPUTDIR):
	mkdir -p $(OUTPUTDIR)
