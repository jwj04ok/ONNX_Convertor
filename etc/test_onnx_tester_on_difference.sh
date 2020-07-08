#!/bin/bash

python optimizer_scripts/onnx_tester.py etc/mobilenet_v2_224.onnx etc/mobilenet_v2_224.cut.onnx
if [ $? -eq 0 ]; then
  echo "Those two model results should be different!"
  exit 1
fi

exit 0
