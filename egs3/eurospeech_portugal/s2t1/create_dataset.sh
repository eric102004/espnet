#!/bin/bash


source path.sh

# We can also create submit.sh for the actual command.
python dataset/create_dataset.py \
    --output_dir data/portugal