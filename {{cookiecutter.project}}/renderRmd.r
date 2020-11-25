#! Rscript

args = commandArgs(trailingOnly=TRUE)
library(rmarkdown)
render(args[1])
