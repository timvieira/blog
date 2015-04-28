PELICAN=pelican
PELICANOPTS=

BASEDIR=$(CURDIR)
INPUTDIR=$(BASEDIR)/content
OUTPUTDIR=$(BASEDIR)/output
CONFFILE=$(BASEDIR)/pelicanconf.py
PUBLISHCONF=$(BASEDIR)/publishconf.py
DEPLOYREPOSITORY=timvieira.github.io

html: clean $(OUTPUTDIR)/index.html
	@echo 'Done'

$(OUTPUTDIR)/%.html:
	$(PELICAN) $(INPUTDIR) -o $(OUTPUTDIR) -s $(CONFFILE) $(PELICANOPTS)

clean:
	find $(OUTPUTDIR) -mindepth 1 -delete

regenerate: clean
	$(PELICAN) -r $(INPUTDIR) -o $(OUTPUTDIR) -s $(CONFFILE) $(PELICANOPTS)

serve:
	cd $(OUTPUTDIR) && gnome-open http://localhost:8000 && python -m SimpleHTTPServer

devserver:
	$(BASEDIR)/develop_server.sh restart

publish:
	$(PELICAN) $(INPUTDIR) -o $(OUTPUTDIR) -s $(PUBLISHCONF) $(PELICANOPTS)

notebook:
	ipython notebook content/notebook/


.PHONY: html help clean regenerate serve devserver publish ssh_upload rsync_upload dropbox_upload ftp_upload github

push:
	( make publish && rsync -a output/. ~/projects/self/timvieira.github.com/blog/. && cd ~/projects/self/timvieira.github.com/ && git add blog && git commit -m 'update blog' && git push )
