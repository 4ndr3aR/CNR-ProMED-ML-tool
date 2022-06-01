#!/bin/bash

if [ ! -z "$1" ] ; then
	echo "Sleeping for $1 seconds..."
	sleep $1
fi

sudo docker build -t cnr-promed-ml-tool . && sudo docker run --rm -it -p 55573:55573 -p 55574:55574 cnr-promed-ml-tool

