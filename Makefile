html: clean
	rm -rf output
	mkdir -p output
	pelican content -s publishconf.py

clean:
	rm -rf output


serve: clean html
	( cd output/ && python -m http.server 8000 )

push: html
	./develop_server.sh stop
	rm -rf ~/projects/self/timvieira.github.com/blog
	mkdir ~/projects/self/timvieira.github.com/blog
	rsync -a output/blog/. ~/projects/self/timvieira.github.com/blog/.
	( cd ~/projects/self/timvieira.github.com/ && hg addremove && hg ci -m 'update blog' && hg bookmarks -r tip master && hg push )
