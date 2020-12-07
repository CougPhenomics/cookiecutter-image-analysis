# Windows User --> If your R_USER directory is not C:\Users\<username>\Documents (which is the default) then you need to change it below
# Required Inputs	: experiment name, start/end dates
EXPER = doi
STARTDATE = 2020-05-31
ENDDATE = 2020-06-05

# Configs
OUTDIR = output
SERVER_CONFIG = $(MY_SERVER_CONFIG) #../cppcserver.config
DATAQUALITY_OUT = reports/PostProcessing.html
DATAQUALITY_SRC = reports/PostProcessing.Rmd
DEVIATION_OUT = reports/deviationheatmaps.html
DEVIATION_SRC = reports/deviationheatmaps.Rmd

VIS_DATADIR = data/vis
VIS_PNG = $(wildcard $(VIS_DATADIR)/*.png)
VIS_OUTDIR = $(OUTDIR)/vis
VIS_OUT = $(VIS_OUTDIR)/vis.json
VIS_CSV_FN = vis.csv
VIS_OUTS_CSV = $(VIS_OUTDIR)/$(VIS_CSV_FN)-single-value-traits.csv
VIS_OUTM_CSV = $(VIS_OUTDIR)/$(VIS_CSV_FN)-multi-value-traits.csv
VIS_CSV1 = $(wildcard $(VIS_OUTDIR)/*traits.csv_level1.csv)
VIS_WORKFLOW = scripts/visworkflow.py

PSII_DATADIR = data/psII
PSII_PNG = $(wildcard $(PSII_DATADIR)/*.png)
PSII_OUTDIR = $(OUTDIR)/psII
PSII_OUT_CSV = $(PSII_OUTDIR)/output_psII_level0.csv
PSII_CSV1 = $(PSII_OUTDIR)/output_psII.csv_level1.csv
PSII_WORKFLOW = scripts/psII.py


PLANTCV_PREFIX = $(realpath $(CONDA_PREFIX))
OS_TYPE = $(shell uname -o)
ifeq ($(OS_TYPE), Msys)
BINDIR = $(PLANTCV_PREFIX)/Scripts
R_USER=$(USERPROFILE)/Documents#<<------- Windows R_USER directory
export R_USER
else
BINDIR = $(PLANTCV_PREFIX)/bin
endif


.PHONY : help
help : Makefile
	# @echo $(PLANTCV_PREFIX)
	@echo Use this file to keep results updated. Required user configuration at top of the file are experiment name - EXPER - and STARTDATE and ENDDATE. See available targets below.
	@sed -n 's/^##//p' $<

## build			: build plantcv docker image
build :
	./build_docker.sh

# use the name of the current directory as the docker image tag
DOCKER_TAG ?= $(shell echo ${PWD} | rev | cut -d/ -f1 | rev)
PROJECT_NAME = $(notdir $(PWD))
DOCKER_USERNAME = dschneiderch
DOCKER_IMAGE = ${DOCKER_USERNAME}/${DOCKER_REPO}:${DOCKER_TAG}

result = $(shell docker ps -f name=$(PROJECT_NAME) -q | wc -l)
## dev				: launch jupyter instance for prototyping and modifying plantcv workflow scripts
dev : build
	@echo $(PROJECT_NAME)
ifeq ($(result),1)
	docker restart $(PROJECT_NAME)
else
	docker run -p 8888:8888 \
		-d \
		--name=$(PROJECT_NAME) \
		--env 'MY_SERVER_CONFIG=$(shell cat ../cppcserver.config)' \
		-v `pwd`/bash_history:/home/jovyan/.bash_history -v `pwd`/data:/home/jovyan/work/data -v `pwd`/scripts:/home/jovyan/work/scripts -v `pwd`/output:/home/jovyan/work/output \
		$(PROJECT_NAME) \
		start.sh jupyter lab --NotebookApp.token='' --NotebookApp.password=''
endif

## shell			: launch bash shell to make full analysis
shell : build
	docker run --rm -it \
		--name=${DOCKER_TAG} \
		-v `pwd`/bash_history:/home/jovyan/.bash_history -v ${PWD}/data:/home/jovyan/work/data -v ${PWD}/scripts:/home/jovyan/work/scripts -v `pwd`/output:/home/jovyan/work/output \
		${DOCKER_TAG} \
		bash


## getvis			: append new VIS images from lemnatec database
# there is no way to to check the filesystem for a dependency but this script will only download new images
.PHONY : getvis
getvis :
	LT_db_extractor \
	--config $(SERVER_CONFIG) \
	--outdir $(VIS_DATADIR) \
	--camera vis \
	--exper $(EXPER) \
	--date1 $(STARTDATE) \
	--date2 $(ENDDATE) \
	--append

## processvis		: run plantcv workflow for vis images
.PHONY : processvis
processvis : $(VIS_OUT) $(VIS_OUTS_CSV) $(VIS_OUTM_CSV)
$(VIS_OUT) : $(VIS_PNG) $(VIS_WORKFLOW)
	mkdir -p $(VIS_OUTDIR)

	ipython $(BINDIR)/plantcv-workflow.py -- \
	--dir data/vis \
	--workflow $(VIS_WORKFLOW) \
	--type png \
	--json $(VIS_OUT) \
	--outdir $(VIS_OUTDIR) \
	--adaptor filename \
	--delimiter "(.{2})-(.+)-(\d{8}T\d{6})-(.+)-(\d)" \
	--timestampformat "%Y%m%dT%H%M%S" \
	--meta plantbarcode,measurementlabel,timestamp,camera,id \
	--cpu 6 \
	--writeimg \
	--create \
	--match plantbarcode:B1 \
	--dates $(STARTDATE)_$(ENDDATE)

# convert json output to csv if needed
$(VIS_OUTS_CSV) $(VIS_OUTM_CSV) : $(VIS_OUT)
	python $(BINDIR)/plantcv-utils.py json2csv -j $(VIS_OUT) -c $(VIS_OUTDIR)/$(VIS_CSV_FN)


## getpsII		: append new PSII images from lemnatec database
.PHONY : getpsII
getpsII :
	LT_db_extractor --config $(SERVER_CONFIG) --outdir $(PSII_DATADIR) --exper $(EXPER) \
	--camera psII \
	--frameid  1 2 \
	--exper $(EXPER) \
	--date1 $(STARTDATE) \
	--date2 $(ENDDATE) \
	--append

## processpsII		: run plantcv workflow for psII images
.PHONY : processpsII
processpsII : $(PSII_OUT_CSV)
$(PSII_OUT_CSV) : $(PSII_PNG) $(PSII_WORKFLOW)
	ipython $(PSII_WORKFLOW)

## dataquality		: Render rmarkdown report of data quality
.PHONY : dataquality
dataquality : $(DATAQUALITY_OUT)
$(DATAQUALITY_OUT) $(PSII_CSV1) $(VIS_CSV1) : $(DATAQUALITY_SRC) $(VIS_OUTS_CSV) $(VIS_OUTM_CSV) $(PSII_OUT_CSV)
	Rscript --vanilla renderRmd.r $<

## wtdeviation		: Render rmarkdown report of deviation from WT
.PHONY : wtdeviation
wtdeviation : $(DEVIATION_OUT)
$(DEVIATION_OUT) : $(DEVIATION_SRC) $(PSII_CSV1) $(VIS_CSV1)
	Rscript --vanilla renderRmd.r $<


## clean-vis		: remove level1 vis csv files
.PHONY : clean-vis
clean-vis :
	rm -f $(VIS_CSV1)

## clean-psII		: remove level1 psII csv files
.PHONY : clean-psII
clean-psII :
	rm -f $(PSII_CSV1)

## clean-processvis	: Remove all output from VIS (image processing output + dataquality csv files)
.PHONY : clean-processvis
clean-processvis : clean-vis
	rm -f $(VIS_OUT)
	rm -f $(VIS_OUTS_CSV)
	rm -f $(VIS_OUTM_CSV)

## clean-processpsII	: Remove all output from PSII (image processing + dataquality csv files)
.PHONY : clean-processpsII
clean-processpsII : clean-psII
	rm -f $(PSII_OUT_CSV)


