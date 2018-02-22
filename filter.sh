#!/usr/bin/env bash


cd src/n01z3
python n01_exif_filter.py
python n02_reduplicate.py
python n03_make_extcsv.py