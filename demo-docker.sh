#!bin/bash

DOCKER_IMG=alphamix
DOCKER_IMG_VERS=v10


# check if image is created
if [[ "$(docker images -q ${DOCKER_IMG}:${DOCKER_IMG_VERS} 2> /dev/null)" == "" ]]; then
  echo "is NOT created"
  nvidia-docker build --no-cache --tag ${DOCKER_IMG}:${DOCKER_IMG_VERS} .
else
  echo "is created"
fi

# Dependencies that are passed as parameters to the docker app
datetime=`date "+%Y%m%d%H%M%S"`
data_name="MNIST"                                                               # DEFINE DATASET
data_dir="your_data_directory/${datetime}"
log_directory="your_log_directory/${datetime}"
n_init_lb=200                                                                   # DEFINE #LABELED SAMPLES
n_query=100                                                                     # DEFINE #QUERIED SAMPLES
n_round=10                                                                      # DEFINE #ROUNDS
learning_rate=0.001
n_epoch=100                                                                     # DEFINE #EPOCHS PER ROUND
model=mlp
strategy=AlphaMixSampling #EntropySampling #RandomSampling                                        # DEFINE QS
alpha_opt

args="--data_name ${data_name} --data_dir ${data_dir} --log_dir ${log_directory} \
        --n_init_lb ${n_init_lb} --n_query ${n_query} --n_round ${n_round} --learning_rate ${learning_rate} --n_epoch ${n_epoch} --model ${model} \
        --strategy ${strategy} --alpha_opt"
echo args


nvidia-docker run --rm -it ${DOCKER_IMG}:${DOCKER_IMG_VERS} ${args}