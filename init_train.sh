#!/usr/bin/env bash
set -e

pushd src
pushd ilya
python init_train.py
popd
popd