#!/bin/bash
# Run with: make build

# back up the environment.yml file just in case you run this script when you don't mean to
mv -v environment.yml "environment.yml.$(date +'%Y%m%dT%H0000')" #2>/dev/null
IMAGE_NAME=Chickpea-disease
DOCKERTAG="$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')" #dockertag must be lowercase
docker build --no-cache -t $DOCKERTAG . #use no-cache to make sure pip install always runs
docker run --rm \
-v `pwd`:/home/jovyan \
--user root -e NB_USER=jovyan -e NB_UID=`id -u` -e NB_GID=`id -g` -e NB_GROUP=`id -gn` \
-e NB_UMASK=002 -e CHOWN_HOME=yes -e CHOWN_HOME_OPTS='-R' \
$DOCKERTAG conda env export --no-builds --file environment.yml --name base #pip freeze > requirements.txt #
# theoretically we are saving the env info so it can be recreated but it doesn't always work

