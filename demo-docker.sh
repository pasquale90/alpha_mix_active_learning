#!bin/bash

########################################################################################## Dependencies that are passed as parameters to the docker app - REQUIRED
datetime=`date "+%Y%m%d%H%M%S"`
data_name="BIRDS" #                                                                # DEFINE DATASET
data_dir="/home/melissap/_Datasets_/BirdsDataset"
output_dir="/home/melissap/Desktop/LAGO/3.githubs/mfork/alpha_mix_active_learning/output"
n_label=525                                                                                         # number of classes
n_training_set=2625                                                                                 # DEFINE #LABELED SAMPLES
n_query=70                                                                                          # DEFINE #QUERIED SAMPLES
n_round=3                                                                                           # DEFINE #ROUNDS
########################################################################################## Dependencies that are passed as parameters to the docker app - OPTIONAL
log_directory="your_log_directory/${datetime}"
learning_rate=0.001
n_epoch=3                                                                     # DEFINE #EPOCHS PER ROUND
model=vit_small
strategy=AlphaMixSampling #EntropySampling #RandomSampling                                        # DEFINE QS
alpha_opt=true

###################################################################################################################################################################

# Parse arguments
args=""
if [ ! -z ${1:-${datetime}} ]; then args="${args} --datetime ${datetime}"; fi
if [ ! -z ${2:-${data_name}} ]; then args="${args} --data_name ${data_name}"; fi
if [ ! -z ${3:-${data_dir}} ]; then args="${args} --data_dir datasets/${data_dir}"; fi
if [ ! -z ${4:-${log_directory}} ]; then args="${args} --log_directory ${log_directory}"; fi
if [ ! -z ${4:-${n_init_lb}} ]; then args="${args} --n_init_lb ${n_training_set}"; fi
if [ ! -z ${6:-${n_query}} ]; then args="${args} --n_query ${n_query}"; fi
if [ ! -z ${7:-${n_round}} ]; then args="${args} --n_round ${n_round}"; fi
if [ ! -z ${8:-${learning_rate}} ]; then args="${args} --learning_rate ${learning_rate}"; fi
if [ ! -z ${9:-${n_epoch}} ]; then args="${args} --n_epoch ${n_epoch}"; fi
if [ ! -z ${10:-${model}} ]; then args="${args} --model ${model}"; fi
if [ ! -z ${11:-${strategy}} ]; then args="${args} --strategy ${strategy}"; fi
if [ ! -z ${12:-${alpha_opt}} ] && ${alpha_opt} ; then args="${args} --alpha_opt"; fi
printf "\n\tbash parameters : \n\t\t${args}\n\n"

################################################################# DOCKERIZATION ####################################################################################
# create docker volumes to share I/O between host and container
DOCKER_VOLUME_IN=${datetime}_alphamix_${data_name}
DOCKER_VOLUME_OUT=${datetime}_alphamix_results
docker volume create --sharing readonly ${DOCKER_VOLUME_IN}
docker volume create --sharing all ${DOCKER_VOLUME_OUT}

# Create a docker image
DOCKER_IMG=alphamix
DOCKER_IMG_VERS=v1
# check if image is created
if [[ "$(docker images -q ${DOCKER_IMG}:${DOCKER_IMG_VERS} 2> /dev/null)" == "" ]]; then
  nvidia-docker build --no-cache --tag ${DOCKER_IMG}:${DOCKER_IMG_VERS} .
fi

# run the container
nvidia-docker run \
      --name=${DOCKER_VOLUME_IN} -v ${data_dir}:/home/alphamix/datasets/${data_dir} \
      --name=${DOCKER_VOLUME_OUT} -v ${output_dir}:/home/alphamix/output \
      --rm -it --gpus all \
      ${DOCKER_IMG}:${DOCKER_IMG_VERS} ${args} /bin/bash

docker volume rm ${DOCKER_VOLUME_IN}
docker volume rm ${DOCKER_VOLUME_OUT}
# clean all
# docker rmi --force $(docker images -q 'alphamix' | uniq) # ------> comment this line if you want the image to be saved
# docker system prune
# nvidia-docker run --rm -it --name devtest exp1 --mount type=volume,source=${data_dir},target=/home/alphamix/datasets/${data_dir} --gpus all ${DOCKER_IMG}:${DOCKER_IMG_VERS} ${args}
###################################################################################################################################################################