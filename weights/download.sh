#!/bin/bash

RELEASE=v1.0.0

curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vits14.onnx -o weights/depth_anything_vits14.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vitb14.onnx -o weights/depth_anything_vitb14.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vitl14.onnx -o weights/depth_anything_vitl14.onnx
