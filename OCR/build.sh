#!/usr/bin/env bash
set -e  # Exit on error

echo "=== Installing Tesseract ==="
sudo apt-get update -y
sudo apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-all

echo "=== Verification ==="
which tesseract
tesseract --version
