now:
	python main.py
	
test:
	rm -rf results/test
	python main.py -f test

process:
	python process.py -i test -f test

fc_experiment:
	python main.py -f exp_fully_connected -b 16 -t 0.01 -o adam -m exp_fc -e 3 -l crossentropy -s 1

lenet_experiment:
	python main.py -f exp_lenet -b 32 -t 0.01 -o adam -m exp_lenet -e 5 -l crossentropy -s 1