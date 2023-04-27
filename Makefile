now:
	python main.py
	
test:
	rm -rf results/main/test
	python main.py -f test -m exp_fc -l crossentropy -e 1

process:
	rm -rf results/process/test
	python process.py -i test -f test -s 10 -u



fc_experiment:
	python main.py -f exp_fc -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist

lenet_experiment:
	python main.py -f exp_lenet -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist 

resnet_experiment:
	python main.py -f exp_resnet -b 256 -t 0.0001 -o adam -m exp_resnet -e 50 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 



fc_process:
	python process.py -i exp_fc -f exp_fc -s 10

lenet_process:
	python process.py -i exp_lenet -f exp_lenet -s 10

resnet_process:
	python process.py -i exp_resnet -f exp_resnet -s 10 -ds cifar10 -S 1 -n 1000



fc_test:
	python test.py -f exp_fc -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 2

lenet_test:
	python test.py -f exp_lenet -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 3

# resnet_process:
# 	python process.py -i exp_resnet -f exp_resnet -s 10 -ds cifar10 -S 1 -n 1000


 
exp_train: fc_experiment lenet_experiment #resnet_experiment

exp_process: fc_process lenet_process #resnet_process

exp: exp_train exp_process



resnet: resnet_experiment resnet_process



test_len:
	rm -rf results/test/test_lenet
	python test.py -f test_lenet -b 32 -t 0.0003 -m exp_lenet -o adam -e 1 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 2