#!/usr/bin/env bash
set -e

mkdir -p data/raw
kaggle datasets download -d ratthachat/writing-prompts -p data/raw --unzip

echo "Dataset downloaded and unzipped into data/raw"
