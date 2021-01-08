#!/bin/bash

git init
git lfs install
git add "*"
git commit -m "Initial file import"
echo "git initial file import complete"
chmod a+x build_docker.sh
