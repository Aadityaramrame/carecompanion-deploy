#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

# Install Tesseract with ALL language packs
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-all
