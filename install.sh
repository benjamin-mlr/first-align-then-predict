#!/usr/bin/env bash

conda create -n align-then-predict python=3.6


conda activate align-then-predict

pip install torch==1.0
pip install transformers==2.8

