html: clean
	python3 build.py

test:
	python3 -m pytest tests/ -v

clean:
	rm -rf output

serve: clean html
	blog dev
