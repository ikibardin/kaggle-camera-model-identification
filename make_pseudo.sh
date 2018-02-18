#!/usr/bin/env bash
set -e

pushd src
pushd ilya
python make_pseudo.py
popd
popd