html: clean
	python3 build.py

clean:
	rm -rf output

serve: clean html
	( cd output/ && python3 -m http.server 8000 )


copy-files: html
	rm -rf ~/projects/self/timvieira.github.com/blog
	mkdir ~/projects/self/timvieira.github.com/blog
	rsync -a output/blog/. ~/projects/self/timvieira.github.com/blog/.

deploy: copy-files
	( cd ~/projects/self/timvieira.github.com/ && git add blog && git commit -m 'update blog' && git push )
