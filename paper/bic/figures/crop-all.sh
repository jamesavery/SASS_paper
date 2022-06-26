#!/bin/bash

for f in *.png; do convert $f -trim +repage $f; done
