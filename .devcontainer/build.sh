#!/bin/bash

docker build -f .devcontainer/Dockerfile  . -t dockerteamcore/transcendence:latest


# docker run  -it --gpus all -v(pwd):/app transcendence:latest bash  

# docker build -f .devcontainer/Dockerfile.gpu  . -t dockerteamcore/transcendence:latest
# docker push  dockerteamcore/transcendence:latest 


