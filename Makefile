test:
	rm -rf results/test
	python main.py -f test
	
now:
	python main.py

process test:
	python process.py -i test -o test