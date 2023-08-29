#!bin/bash

datetime=`date "+%Y%m%d%H%M%S"`
data_name="MNIST" #"BIRDS"                                                               # DEFINE DATASET
data_dir="/home/melissap/Desktop/LAGO/3.githubs/mfork/alpha_mix_active_learning/your_data_directory/BirdsDataset" #"your_data_directory/${datetime}"
output_dir="/home/melissap/Desktop/LAGO/3.githubs/mfork/alpha_mix_active_learning/output"
log_directory="your_log_directory/${datetime}"
n_init_lb=100 #2625                                                                   # DEFINE #LABELED SAMPLES
n_query=100                                                                     # DEFINE #QUERIED SAMPLES
n_round=10 #5                                                                      # DEFINE #ROUNDS
learning_rate=0.001
n_epoch=100 #5                                                                     # DEFINE #EPOCHS PER ROUND
model=mlp #vit_small
strategy=AlphaMixSampling #EntropySampling #RandomSampling                                        # DEFINE QS
alpha_opt

args="--data_name ${data_name} --data_dir ${data_dir} --log_dir ${log_directory} \
        --n_init_lb ${n_init_lb} --n_query ${n_query} --n_round ${n_round} --learning_rate ${learning_rate} --n_epoch ${n_epoch} --model ${model} \
        --strategy ${strategy} --alpha_opt"
echo args

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alphamix

mkdir ${log_directory}
# python main.py ${args} &> ${log_directory}/main.log
python main.py ${args}
