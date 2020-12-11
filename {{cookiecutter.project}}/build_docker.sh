#!/bin/bash
# Run with: make build

# back up the environment.yml file just in case you run this script when you don't mean to
mv -v environment.yml "environment.yml.$(date +'%Y%m%dT%H%M%S')" #2>/dev/null
IMAGE_NAME={{ cookiecutter.project }}
DOCKERTAG="$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')" #dockertag must be lowercase
docker build --no-cache -t $DOCKERTAG . #use no-cache to make sure pip install always runs
docker run --rm -v `pwd`:/home/jovyan $DOCKERTAG conda env export --no-builds --file environment.yml --name base #pip freeze > requirements.txt #
# theoretically we are saving the env info so it can be recreated but it doesn't always work
