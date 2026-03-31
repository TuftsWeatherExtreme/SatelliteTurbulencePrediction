#!/bin/bash

# unload_modules.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Unload all modules and deactivate environment.

echo "Deactivating conda environment (if any)"
conda deactivate 2>/dev/null

echo "Purging all modules"
module purge 2>/dev/null

echo "Done! Environment has been torn down"
