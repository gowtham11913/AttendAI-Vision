# Third-Party Notices

## Anti-spoofing models (models/anti_spoof_models/)

This project includes two ONNX models converted from pretrained weights
published by the **Silent-Face-Anti-Spoofing** project:

    Repository: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    Author:     Minivision AI (小视科技)
    License:    Apache License 2.0 (full text in this directory:
                LICENSE_Silent-Face-Anti-Spoofing.txt)

Files in this project:
    models/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx
    models/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.onnx

These were converted from the original PyTorch (.pth) weights to ONNX
format using `torch.onnx.export`, with numerical equivalence verified
against the original PyTorch model output (max absolute difference
~1e-6 on identical input). No retraining was performed - these are the
original published weights, just in a different serialization format so
this project can run them via onnxruntime without requiring PyTorch as a
runtime dependency.

The inference logic in `utils/antispoof.py` (preprocessing, scale-crop
expansion, model ensembling, and real/fake classification convention) was
reimplemented to match the original repository's `src/anti_spoof_predict.py`
and `src/generate_patches.py`, adapted to use this project's YuNet face
detector instead of the original repo's bundled RetinaFace Caffe model.

If you redistribute this project, please retain this notice and the
accompanying LICENSE_Silent-Face-Anti-Spoofing.txt file, per the terms of
the Apache License 2.0.
