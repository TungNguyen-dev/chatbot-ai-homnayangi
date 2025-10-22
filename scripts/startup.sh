#!/bin/bash/env bash

# 0. Remove .venv
rm -rf .venv

# 1. Create and activate environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install project dependencies
pip install -r requirements.txt

mkdir -p storage