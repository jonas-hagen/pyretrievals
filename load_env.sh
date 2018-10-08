#!/usr/bin/env bash
#Load relevant variables from .env file

ENV_FILE=".env"

source $ENV_FILE
export ARTS_INCLUDE_PATH=$ARTS_HOME/arts/controlfiles/
export ARTS_BUILD_PATH=$ARTS_HOME/arts/build/
export ARTS_DATA_PATH=$ARTS_HOME/arts-xml-data/
export ARTS_SRC_PATH=$ARTS_HOME/arts/