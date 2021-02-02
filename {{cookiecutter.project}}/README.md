# {{ cookiecutter.project }}

## Introduction

This image analysis project has been setup to take advantage of docker containers for image processing. You will need to install [Docker Desktop](https://www.docker.com/products/docker-desktop) if you haven't already.

This assumes you are at a terminal and after creating the cookiecutter project you moved into the new project directory [Chickpea-disease] with `cd Chickpea-disease` and now you should see the following file structure. In the directory directly above the project folder you should have a valid `cppcserver.config` file with your server credentials. 

```
(base) [dominik@cppc-server Chickpea-disease]$ tree
.
├── build_docker.sh
├── Dockerfile
├── download_data.sh
├── LICENSE.md
├── Makefile
├── output
├── plantcv.Makefile
├── project.yml
├── README.md
├── renderRmd.r
├── reports
│   ├── deviationheatmaps.Rmd
│   └── PostProcessing.Rmd
├── run_workflows.sh
└── scripts
    ├── psII.py
    └── visworkflow.py

3 directories, 14 files
```

## Setting up Docker

There are three `make` commands to work with docker.  You can see the help directly by running `make`
```
(base) [dominik@cppc-server Chickpea-disease]$ make
Use this file to simplify docker interaction. See available targets below. docker ps -a to see containers.
 build                  : build plantcv docker image
 dev                    : launch jupyter instance for prototyping and modifying plantcv workflow scripts
 shell                  : launch separate bash shell for running container
 ```

Use `make build` to build the docker image. You only need to do this once. 

Use `make dev` to launch a docker container running jupyter. Then you can use the browser to access the jupyter lab workspace on port 8888. You can either run it locally or on the server. Locally you would access it from `localhost:8888`. If you are using the server then go to `ipaddress:8888` where ipaddress is the address of the server. If you are off campus you need to be connected with a VPN. 

Use `make shell` to launch a separate bash shell for the running container.

## Getting Started with the Analysis

Once you are in Jupyter Lab, you need to go to `work` directory with
```
cd work
```
Here you will find directories

```
(base) jovyan@8a5340091f9d:~/work$ tree -d
.
├── data
├── output
├── reports
└── scripts

```

See the help for running different steps within the workflow: 

```
(base) jovyan@8a5340091f9d:~/work$ make
Use this file to keep results updated. Required user configuration at top of the file are experiment name - EXPER - and STARTDATE and ENDDATE. See available targets below.
 getvis                 : append new VIS images from lemnatec database
 processvis             : run plantcv workflow for vis images
 getpsII                : append new PSII images from lemnatec database
 processpsII            : run plantcv workflow for psII images
 dataquality            : Render rmarkdown report of data quality
 wtdeviation            : Render rmarkdown report of deviation from WT
 clean-vis              : remove level1 vis csv files
 clean-psII             : remove level1 psII csv files
 clean-processvis       : Remove all output from VIS (image processing output + dataquality csv files)
 clean-processpsII      : Remove all output from PSII (image processing + dataquality csv files)
```

### Getting Data

`make getvis` will download all available RGB images but you can modify the Makefile in Docker container accordingly.

`make getpsII` will download all available PSII images but you can modify the Makefile in Docker container accordingly.

### Analysis scripts

You probably need to tweak the analysis scripts for peculiarities in your dataset, particularly pertaining to the image segmentation. The scripts are found in `scripts/` directory. Open these in JupyterLab and test the segmentation on a few images.  When you are ready you can run the scripts over all images with 

`make processvis` and `make processpsII`

### Post-Analysis Reports

You can check the results of the image analysis project by generating RMarkdown reports using `make dataquality` and `make wtdeviation`


