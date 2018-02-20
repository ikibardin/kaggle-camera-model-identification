#!/usr/bin/env bash
set -e

pushd src
pushd ilya
python final_train.py
popd
popd