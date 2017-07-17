# get code
sudo rm -r ~/code/cnn_workshop
git clone https://github.com/marco-willi/cnn_workshop.git ~/code/cnn_workshop

# run nvidia docker with jupyter notebook
# restrict access to AWS instance to SSH
# and TCP custom, my IP and port 8888
# this is a custom-compiled tensorflow docker, you can also use the container
# published on docker hub
sudo nvidia-docker run -it -p 8888:8888 -v ~/:/host root/tensorflow:latest-devel-gpu-py3 jupyter notebook --allow-root --notebook-dir=/host/code/cnn_workshop



# run
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash


sudo docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# normal docker with jupyter access
docker run -it --rm -p 8888:8888 jupyter/tensorflow-notebook

sudo docker run -it -p 8888:8888 root/tensorflow:latest-devel-gpu-py3 jupyter notebook

# run normal docker with jupyter notebook
# restrict access to AWS instance to SSH
# and TCP custom, my IP and port 8888
sudo docker run -it -p 8888:8888 -v ~/:/host root/tensorflow:latest-devel-gpu-py3 jupyter notebook --allow-root --notebook-dir=/host/code/cnn_workshop
sudo nvidia-docker run -it -p 8888:8888 -v ~/:/host root/tensorflow:latest-devel-gpu-py3 jupyter notebook --allow-root --notebook-dir=/host/code/cnn_workshop


# non GPU docker
sudo docker run -it -p 8888:8888 root/tensorflow:latest-devel-gpu-py3 jupyter notebook --allow-root

jupyter notebook --notebook-dir=/absolute/path/to/notebook/directory
