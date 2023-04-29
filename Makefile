now:
	python main.py
	
test:
	rm -rf results/main/test
	python main.py -f test -m exp_fc -l crossentropy -e 2

process:
	rm -rf results/process/test
	python process.py -i test -f test -s 10 -u

test_test:
	rm -rf results/test/test
	python test.py -f test -m exp_fc -l crossentropy -e 2 -fp 2 -i test



fc_experiment:
	python main.py -f exp_fc -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist

lenet_experiment:
	python main.py -f exp_lenet -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist 

resnet_experiment:
	python main.py -f exp_resnet -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda



fc_process:
	python process.py -i exp_fc -f exp_fc -s 10

lenet_process:
	python process.py -i exp_lenet -f exp_lenet -s 10

resnet_process:
	python process.py -i exp_resnet -f exp_resnet -s 10 -ds cifar10 -S 1 -n 1000



fc_test:
	python test.py -f exp_fc_1 -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 1
	python test.py -f exp_fc_2 -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 2
	python test.py -f exp_fc_3 -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 3

lenet_test:
	python test.py -f exp_lenet_1 -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 1
	python test.py -f exp_lenet_2 -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 2
	python test.py -f exp_lenet_3 -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 3
	python test.py -f exp_lenet_4 -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 4
	python test.py -f exp_lenet_5 -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist -fp 5

resnet_test:
	python test.py -f exp_resnet_1 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 1
	python test.py -f exp_resnet_10 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 10 -i exp_resnet
	python test.py -f exp_resnet_20 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 20 -i exp_resnet
	python test.py -f exp_resnet_30 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 30 -i exp_resnet
	python test.py -f exp_resnet_40 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 40 -i exp_resnet
	python test.py -f exp_resnet_50 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 50 -i exp_resnet
	python test.py -f exp_resnet_60 -b 256 -t 0.0001 -o adam -m exp_resnet -e 70 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d cuda -fp 60 -i exp_resnet

 
exp_train: fc_experiment lenet_experiment #resnet_experiment

exp_process: fc_process lenet_process #resnet_process

exp_test: fc_test lenet_test

exp: exp_process exp_test


resnet: resnet_experiment resnet_test



test_len:
	rm -rf results/test/test_lenet
	python test.py -f test_lenet -b 32 -t 0.0003 -m exp_lenet -o adam -e 3 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 2

test_len2:
	rm -rf results/main/test_lenet
	python main.py -f test_lenet -b 32 -t 0.0003 -m exp_lenet -o adam -e 3 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist