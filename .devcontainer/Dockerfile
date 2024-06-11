FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime


USER root
# Set the working directory
WORKDIR /


# install dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends -q git g++ python3-opengl openssh-client xvfb vim fish wget curl  libgl1 software-properties-common openssh-server nginx

RUN usermod --shell /usr/bin/fish root

RUN conda create -n py311 python=3.11 --yes
RUN conda init bash
RUN echo "source activate py311" >> ~/.bashrc

RUN conda init fish
RUN mkdir -p ~/.config/fish
RUN echo "conda activate py311" >> ~/.config/fish/config.fish
ENV PATH /opt/conda/envs/py311/bin:$PATH


COPY requirements.txt .
RUN pip install -r requirements.txt



# Set up Jupyter Notebook
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension



# NGINX Proxy
COPY .devcontainer/nginx.conf /etc/nginx/nginx.conf
COPY .devcontainer/readme.html /usr/share/nginx/html/readme.html


# Start Scripts
COPY .devcontainer/start.sh /
RUN chmod +x /start.sh


# Set the default command for the container
CMD [ "/start.sh" ]

