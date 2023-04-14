now:
	python main.py
	
test:
	rm -rf results/test
	python main.py -f test -m exp_fc -l crossentropy

process:
	rm -rf processed/test
	python process.py -i test -f test -s 10 -u

fc_experiment:
	python main.py -f exp_fc -b 32 -t 0.0003 -o adam -m exp_fc -e 3 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist

lenet_experiment:
	python main.py -f exp_lenet -b 32 -t 0.0003 -o adam -m exp_lenet -e 5 -l crossentropy -s 1 -w 0.000001 -sr 10 -ds mnist

resnet_experiment:
	python main.py -f exp_resnet -b 256 -t 0.0001 -o adam -m exp_resnet -e 50 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10 -d='cuda:1'

test_res:
	rm -rf results/exp_resnet
	python main.py -f exp_resnet -b 256 -t 0.0001 -o adam -m exp_resnet -e 50 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10

fc_process:
	python process.py -i exp_fc -f exp_fc -s 10

lenet_process:
	python process.py -i exp_lenet -f exp_lenet -s 10

resnet_process:
	python process.py -i exp_resnet -f exp_resnet -s 10 -ds cifar10 -S 1 -n 5000

exp_train: fc_experiment lenet_experiment

exp_process: fc_process lenet_process

resnet: resnet_experiment resnet_process

fc_exp_test:
	rm -rf processed/exp_fc_t
	python process.py -i exp_fc -f exp_fc_t -s 10

lenet_exp_test:
	rm -rf processed/exp_lenet
	python process.py -i exp_lenet -f exp_lenet -s 10

exp: exp_train exp_process

test_len:
	rm -rf test_results/test_lenet
	python test.py -f test_lenet -b 32 -t 0.0003 -o adam -e 1 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 1

run_test_exp:
	python test.py -f fc_plain -b 32 -t 0.0003 -o adam -e 3 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 0
	python test.py -f fc_fp_2 -b 32 -t 0.0003 -o adam -e 3 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 2
	python test.py -f fc_fp_3 -b 32 -t 0.0003 -o adam -e 3 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds mnist -fp 3


test_res_1:
	rm -rf results/exp_resnet_1_epoch
	python main.py -f exp_resnet_1_epoch -b 256 -t 0.0001 -o adam -m exp_resnet -e 1 -l crossentropy -s 1 -w 0.00001 -sr 10 -ds cifar10