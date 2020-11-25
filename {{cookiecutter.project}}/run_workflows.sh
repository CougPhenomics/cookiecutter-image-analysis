#!
if [ `uname -o` ==  'Msys' ]
then
bindir=$CONDA_PREFIX/Scripts
else
bindir=$CONDA_PREFIX/bin
fi

# echo $bindir

outdir=output/vis
mkdir -p $outdir

python $bindir/plantcv-workflow.py \
--dir data/vis \
--workflow scripts/visworkflow.py \
--type png \
--json $outdir/vis.test.json \
--outdir $outdir \
--adaptor filename \
--delimiter "(.{2})-(.+)-(\d{8}T\d{6})-(.+)-(\d+)" \
--timestampformat "%Y%m%dT%H%M%S" \
--meta plantbarcode,measurementlabel,timestamp,camera,id \
--cpu 6 \
--writeimg \
--create \
--dates 2020-05-31_2020-06-01 \
--match plantbarcode:A2,camera:VIS0
# --other_args="--pdfs data/naive_bayes_training/naive_bayes_pdfs.tsv"

# looks like all vis images on 2020-06-17 are blank

python $bindir/plantcv-utils.py json2csv -j $outdir/vis.test.json -c $outdir/vis.test.csv

