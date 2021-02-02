# Cookiecutter Phenomics project template

This repo contains the project template for phenomics experiment at Compact Plants Phenomics Center at Washington State University - Pullman. It includes scripts to download images from the server, analyze them, and produce a data quality report. A python environment is setup in a Docker container for downloading and analyzing images. Different targets of the image analysis workflow are defined in a Makefile to minimize the number of commands that need to be memorized.

## Getting Started

You need to have git and the cookiecutter python package installed to run the command. You will need to have Docker installed to make use of the tools.

Then run the following command in the terminal at the parent directory to download and setup the latest project files. When it asks for the project name you should be the experiment name exactly (including capitalization) as it was called in the LemnaTec software because this will be used to download the image files.

```
cookiecutter gh:CougPhenomics/cookiecutter-phenomics
```

See the readme in the project directory for detailed documentation to use the files (also available in [{{cookiecutter.project}}](cookiecutter.project/README.md))



## Dependencies

- git
- python>=3.6
- [Cookiecutter](http://cookiecutter.readthedocs.io/) version 1.4 or greater.
- [Docker](https://www.docker.com/products/docker-desktop)
- R

Additionally, the Rmarkdown scripts will require a local R installation that can load

```
library(here)
library(tidyverse)
library(cppcutils) # available with remotes::install_github('cougphenomics/cppcutils')
library(knitr)  
require(xtable)
require(lubridate)
require(assertthat)
```

The goal is to include a Dockerized solution for the R portion too.

## Credits

This project was originally forked from https://github.com/JIC-Image-Analysis/cookiecutter-image-analysis
