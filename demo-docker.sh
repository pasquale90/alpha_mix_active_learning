#!bin/bash

DOCKER_IMG=alphamix
DOCKER_IMG_VERS=v1

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
alpha_opt=true

# Parse arguments
args=""
if [ ! -z ${1:-${datetime}} ]; then args="${args} --datetime ${datetime}"; fi
if [ ! -z ${2:-${data_name}} ]; then args="${args} --data_name ${data_name}"; fi
if [ ! -z ${3:-${data_dir}} ]; then args="${args} --data_dir ${data_dir}"; fi
if [ ! -z ${4:-${log_directory}} ]; then args="${args} --log_directory ${log_directory}"; fi
if [ ! -z ${4:-${n_init_lb}} ]; then args="${args} --n_init_lb ${n_init_lb}"; fi
if [ ! -z ${6:-${n_query}} ]; then args="${args} --n_query ${n_query}"; fi
if [ ! -z ${7:-${n_round}} ]; then args="${args} --n_round ${n_round}"; fi
if [ ! -z ${8:-${learning_rate}} ]; then args="${args} --learning_rate ${learning_rate}"; fi
if [ ! -z ${9:-${n_epoch}} ]; then args="${args} --n_epoch ${n_epoch}"; fi
if [ ! -z ${10:-${model}} ]; then args="${args} --model ${model}"; fi
if [ ! -z ${11:-${strategy}} ]; then args="${args} --strategy ${strategy}"; fi
if [ ! -z ${12:-${alpha_opt}} ] && ${alpha_opt} ; then args="${args} --alpha_opt"; fi

printf "\n\tbash parameters : \n\t\t${args}\n\n"

# check if image is created
if [[ "$(docker images -q ${DOCKER_IMG}:${DOCKER_IMG_VERS} 2> /dev/null)" == "" ]]; then
  nvidia-docker build --no-cache --tag ${DOCKER_IMG}:${DOCKER_IMG_VERS} .
fi

nvidia-docker run --rm -it --gpus all ${DOCKER_IMG}:${DOCKER_IMG_VERS} ${args}
