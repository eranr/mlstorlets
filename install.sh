git clone https://github.com/openstack/storlets ~/storlets
cd ~/storlets
./s2aio.sh dev host

NUM_IMAGES=`sudo docker images | grep  -v ubuntu | grep -v REPOSITORY | awk '{print $1}' | wc -l`

if [ $NUM_IMAGES != 1 ]; then
    echo "Cannot determine the project id. Please execute install.sh [project id]"
    exit
fi

PROJECT_ID=`sudo docker images | grep  -v ubuntu | grep -v REPOSITORY | awk '{print $1}'`

mkdir -p /tmp/update_docker_image
cat <<EOF >/tmp/update_docker_image/Dockerfile
FROM $PROJECT_ID

MAINTAINER root

# The following operations should be defined in one line
# to prevent docker images from including apt cache file.
RUN apt-get update && \
    apt-get install python-numpy -y && \
    apt-get install python-scipy -y && \
    pip install scikit-learn
EOF

sudo docker build -t storlet_scikit_learn /tmp/update_docker_image
IMAGE_ID=`sudo docker images | grep storlet_scikit_learn | awk '{print $3}'`
sudo docker rmi $PROJECT_ID
sudo docker tag $IMAGE_ID $PROJECT_ID
sudo docker rmi storlet_scikit_learn

rm -fr /tmp/update_docker_image
