

--------------------------------模型转化----------------
环境安装
pip install onnx-simplifier
conda install onnx



---------------------- RT 个人操作版本 -----------------

1、pt 转化为onnx  (直接跑可能跑不起来，原本是在fastreid环境下跑的)
python 02_py2onnx.py --config-file ./configs/Market1501/bagtricks_R101-ibn.yml MODEL.WEIGHTS ../market_bot_R101-ibn.pth MODEL.DEVICE "cuda:0"

2、onnx 转成rt 并实现模型推理

python 03_onnx2rt.py










（京东原版）


Run onnx_export.py to get the converted ONNX model,

在 deplay文件夹下运行
1 、 python onnx_export.py --config-file ../../configs/Market1501/bagtricks_R101-ibn.yml --name "baseline_R101" --output /data/disk1/project/model_dump/01_reid --opts MODEL.WEIGHTS /data/disk1/project/model_dump/01_reid/market_bot_R101-ibn.pth

then you can check the ONNX model in outputs/onnx_model.






Run onnx_inference.py to save ONNX model features with input images

 python onnx_inference.py --model-path /data/disk1/project/model_dump/01_reid/baseline_R101.onnx \
 --input test_data/*.jpg --output /data/disk1/project/model_dump/01_reid/onnx_output



python demo.py --config-file ../configs/Market1501/bagtricks_R101-ibn.yml --input ../tools/deploy/test_data/*.jpg --output /data/disk1/project/model_dump/01_reid

np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-6)






2、将onnx 转化为RT模型

python trt_export.py --name "baseline_R101" --output /data/disk1/project/model_dump/01_reid --onnx-model /data/disk1/project/model_dump/01_reid/baseline_R101.onnx --height 256 --width 128

then you can check the TRT model in outputs/trt_model.

Run trt_inference.py to save TRT model features with input images



 python onnx_inference.py --model-path outputs/trt_model/baseline.engine \
 --input test_data/*.jpg --output trt_output --output-name trt_model_outputname

Run demo/demo.py to get fastreid model features with the same input images, then verify that TensorRT and PyTorch are computing the same value for the network.



np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-6)
