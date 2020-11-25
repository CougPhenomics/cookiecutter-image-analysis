#! /bin/bash

ipython /home/dominik/Documents/data-science-tools/LT-db-extractor.py -- \
--config ../cppcserver-local.config \
--outdir data/psII \
--camera psII \
--exper doi \
--append

ipython /home/dominik/Documents/data-science-tools/LT-db-extractor.py -- \
--config ../cppcserver-local.config \
--outdir data/vis \
--camera vis \
--exper doi \
--append
