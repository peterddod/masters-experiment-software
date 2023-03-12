now:
	python main.py
	
test:
	rm -rf results/test
	python main.py -f test -m exp_fc -l crossentropy

process:
	rm -rf processed/test
	python file_process.py -i test -f test -s 10

fc_experiment:
	python main.py -f exp_fc -b 16 -t 0.01 -o adam -m exp_fc -e 3 -l crossentropy -s 1

lenet_experiment:
	python main.py -f exp_lenet -b 32 -t 0.01 -o adam -m exp_lenet -e 5 -l crossentropy -s 1

fc_process:
	python file_process.py -i exp_fc -f exp_fc -s 10

lenet_process:
	python file_process.py -i exp_lenet -f exp_lenet -s 10

exp_train: fc_experiment lenet_experiment

exp_process: fc_process lenet_process