#!bin/bash

datetime=`date "+%Y%m%d%H%M%S"`
data_name="MNIST"                                                               # DEFINE DATASET
data_dir="/home/melissap/_Datasets_/BirdsDataset/" #"your_data_directory/${datetime}"
output_dir="/home/melissap/Desktop/LAGO/3.githubs/mfork/alpha_mix_active_learning/_output"
log_directory="your_log_directory/${datetime}"
n_init_lb=100                                                                   # DEFINE #LABELED SAMPLES
n_query=100                                                                    # DEFINE #QUERIED SAMPLES 
curr_round=1
n_round=5                                                                      # DEFINE #ROUNDS
learning_rate=0.001
n_epoch=100                                                                     # DEFINE #EPOCHS PER ROUND
model=mlp #vit_small
strategy=AlphaMixSampling #EntropySampling #RandomSampling                                        # DEFINE QS
alpha_opt

args="--data_name ${data_name} --data_dir ${data_dir} --log_dir ${log_directory} \
        --n_init_lb ${n_init_lb} --n_query ${n_query} --curr_round ${curr_round} --n_round ${n_round} --learning_rate ${learning_rate} --n_epoch ${n_epoch} --model ${model} \
        --strategy ${strategy} --alpha_opt"
echo ${args}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate customLearner3

# python customLearner_main.py ${args} #&> _results.log

# python main.py --data_name MNIST --data_dir your_data_directory --log_dir your_log_directory --n_init_lb 100 --n_query 100 --n_round 10 --learning_rate 0.001 --n_epoch 1000 --model mlp --strategy AlphaMixSampling --alpha_opt