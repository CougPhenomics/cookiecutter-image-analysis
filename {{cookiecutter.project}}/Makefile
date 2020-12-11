
# use the name of the current directory as the docker image tag
PROJECT_NAME = $(notdir $(PWD))
DOCKERTAG = $(shell echo $(PROJECT_NAME) | tr '[:upper:]' '[:lower:]') #LOWERCASE DOCKERTAG
container_running = $(shell docker ps -f name=$(DOCKERTAG) -f status=running -q) # see if there are running containers rather than waiting for build and fail
container_exists = $(shell docker ps -a -f name=$(DOCKERTAG) -q) # see if there are running containers rather than waiting for build and fail


.PHONY : help
help : Makefile
	@echo Use this file to simplify docker interaction. See available targets below. 'docker ps -a' to see containers.
	@sed -n 's/^##//p' $<


## build			: build plantcv docker image
.PHONY : build
build :
	@echo Building $(PROJECT_NAME)...
ifeq ($(strip $(container_running)),) # don't build if container is running but rebuild if a container is exited
	./build_docker.sh
else
	@echo "! Build script not run because a container with $(PROJECT_NAME) is running."
endif


## dev			: launch jupyter instance for prototyping and modifying plantcv workflow scripts
.PHONY : dev
dev :
ifeq ($(strip $(container_exists)),) # don't launch if container exists
	docker run -p 8888:8888 \
		-d \
		--name=$(DOCKERTAG) \
		-v `pwd`/data:/home/jovyan/work/data -v `pwd`/scripts:/home/jovyan/work/scripts -v `pwd`/output:/home/jovyan/work/output \
		$(DOCKERTAG) \
		start.sh jupyter lab --NotebookApp.token='' --NotebookApp.password=''
	docker cp ../cppcserver.config $(DOCKERTAG):/home/jovyan/
else
	@echo "There is already a container named $(DOCKERTAG). Investigate with 'docker ps -a $(DOCKERTAG)'. Use 'docker start $(DOCKERTAG)' to restart it. Go to localhost:8888/lab if it's already running."
endif


## shell			: launch separate bash shell for running container
.PHONY : shell
shell :
ifneq ($(strip $(container_running)),) # don't launch if container exists
	docker exec -it `docker ps -f name=$(DOCKERTAG) -q` /bin/bash
else
	@echo "There is no running container. Run 'make dev' first."
endif
