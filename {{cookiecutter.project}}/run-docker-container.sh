#!/bin/bash

CONTAINER="{{ cookiecutter.project }}"
touch `pwd`/bash_history
docker run -it --rm -p 8888:8888 -v `pwd`/bash_history:/home/jovyan/.bash_history -v `pwd`/data:/home/jovyan/work/data:ro -v `pwd`/scripts:/home/jovyan/work/scripts:ro -v `pwd`/output:/home/jovyan/work/output $CONTAINER
