#!/bin/env bash

if [ ! -d "./env" ]; then
	echo "env does not exist"
	echo "creating env"
	python3.11 -m venv env 
	echo "env created"
	echo ""
fi

source ./env/bin/activate
python3.11 -m pip install pip --upgrade
python3.11 -m pip install poetry
python3.11 -m poetry install
