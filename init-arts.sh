#!/bin/bash

if [ -z $ARTS_DATA_PATH ]; then echo "Error: Environment variable ARTS_DATA_PATH not set."; exit 1; fi
if [ -z $ARTS_SRC_PATH ]; then echo "Error: Environment variable ARTS_SRC_PATH not set."; exit 1; fi

# Update the svn working copy
svn co https://arts.mi.uni-hamburg.de/svn/rt/arts/trunk $(realpath ${ARTS_SRC_PATH})
svn co https://arts.mi.uni-hamburg.de/svn/rt/arts-xml-data/trunk/ $(realpath ${ARTS_DATA_PATH})

echo 'Done.'

