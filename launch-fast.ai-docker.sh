#!/bin/bash

if [ ! -z "$1" ] ; then
	echo "Sleeping for $1 seconds..."
	sleep $1
fi

sudo docker build -t cnr-cable-problem-ml-model . && sudo docker run --rm -it -p 55563:55563 -p 55564:55564 cnr-cable-problem-ml-model

