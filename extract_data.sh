#!/bin/bash

# Script to extract split zip files for diabetic retinopathy dataset
# Usage: ./extract_data.sh

echo "Combining split zip files..."
cat train.zip.* > train_combined.zip

echo "Extracting combined archive..."
unzip train_combined.zip -d data/

echo "Cleaning up combined zip file..."
rm train_combined.zip

echo "Extraction complete!"
echo "Data should now be in: data/train/ (images) and data/trainLabels.csv (labels)"
echo "You can now run the notebook with the updated paths."