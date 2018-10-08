#!/usr/bin/env bash
# Creates a .env file and exports the variables.

ENV_FILE=".env"

if [ -z $ARTS_HOME ]; then
    echo "Error: Environment variable ARTS_HOME not set (example: /dev/arts).";
else
    echo "# Auto created by create_env.sh" > $ENV_FILE
    echo "ARTS_HOME=$ARTS_HOME" >> $ENV_FILE
    echo "ARTS_INCLUDE_PATH=$ARTS_HOME/arts/controlfiles/" >> $ENV_FILE
    echo "ARTS_DATA_PATH=$ARTS_HOME/arts-xml-data/" >> $ENV_FILE
    echo "ARTS_BUILD_PATH=$ARTS_HOME/arts/build/" >> $ENV_FILE
    echo "ARTS_SRC_PATH=$ARTS_HOME/arts/" >> $ENV_FILE
fi

