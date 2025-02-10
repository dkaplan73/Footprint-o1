#!/bin/bash
echo "------------------------------"
echo "Installing required packages..."
pip install --upgrade pip
pip install tensorflow opencv-python albumentations numpy matplotlib scikit-learn tf2onnx onnx onnxruntime

echo "------------------------------"
echo "Running dataset_loader.py (loads images and saves dataset.npz)"
python dataset_loader.py

echo "------------------------------"
echo "Running augment_data.py (augments data and saves dataset_augmented.npz)"
python augment_data.py

echo "------------------------------"
echo "Running train_improved.py (training on augmented data)"
python train_improved.py

echo "------------------------------"
echo "Running fine_tune.py (fine-tuning using original data)"
python fine_tune.py

echo "------------------------------"
echo "Running evaluate.py (evaluating model)"
python evaluate.py

echo "------------------------------"
echo "Running postprocess.py (post-processing predictions)"
python postprocess.py

echo "------------------------------"
echo "Converting model to ONNX format..."
python convert_to_onnx.py

echo "------------------------------"
echo "Testing ONNX model..."
python onnx_test.py

echo "------------------------------"
echo "Converting model to TFLite format..."
python deploy.py

echo "------------------------------"
echo "Running predict.py (displaying a sample prediction)"
python predict.py

echo "------------------------------"
echo "All steps completed successfully."

