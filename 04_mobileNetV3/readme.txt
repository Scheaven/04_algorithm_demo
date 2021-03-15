python main.py

推理：
python predict.py --imgs "./data/"

转化onnx模型
python onnx_export.py
onnx模型的推理（好像和原来的不同）
python onnx_inference.py --input test_data/*.jpg  --model_path ./onnx_model/baseline.engine


转化trt模型
python trt_export.py --name "baseline" --output ./onnx_model --onnx-model ./onnx_model/baseline.onnx


