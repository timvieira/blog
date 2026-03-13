html: clean
	python3 build.py

clean:
	rm -rf output

serve: clean html
	( cd output/blog/ && python3 -m http.server 8000 )

deploy: html
	rm -rf ~/projects/self/timvieira.github.com/blog2
	mkdir ~/projects/self/timvieira.github.com/blog2
	rsync -a output/blog/. ~/projects/self/timvieira.github.com/blog2/.
	( cd ~/projects/self/timvieira.github.com/ && git add blog2 && git commit -m 'update blog' && git push )
