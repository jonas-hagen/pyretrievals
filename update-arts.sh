#!/bin/bash

if [ -z $ARTS_DATA_PATH ]; then echo "Error: Environment variable ARTS_DATA_PATH not set."; exit 1; fi
if [ -z $ARTS_BUILD_PATH ]; then echo "Error: Environment variable ARTS_BUILS_PATH not set."; exit 1; fi
if [ -z $ARTS_SRC_PATH ]; then echo "Error: Environment variable ARTS_SRC_PATH not set."; exit 1; fi

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Update the svn working copy
cd $ARTS_SRC_PATH
svn revert --recursive .
svn update

cd $ARTS_DATA_PATH
svn revert --recursive .
svn update

# Compile arts
cd $ARTS_BUILD_PATH
cmake -DENABLE_C_API=1 -DARTS_XML_DATA_PATH=$ARTS_DATA_PATH -DCMAKE_BUILD_TYPE=RelWithDebInfo $ARTS_SRC_PATH
make clean
make -j5

echo 'Done.'

