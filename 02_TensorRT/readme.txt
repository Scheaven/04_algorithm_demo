---------------------- RT 个人操作版本 -----------------



1、pt 转化为onnx  (直接跑可能跑不起来，原本是在京东fastreid环境下跑的)
python 02_py2onnx.py --config-file ./configs/Market1501/bagtricks_R101-ibn.yml MODEL.WEIGHTS ../market_bot_R101-ibn.pth MODEL.DEVICE "cuda:0"

01_onnx_export.py 是根据京东代码改造的原本生成onnx格式的文件


2、onnx 转成rt 并实现模型推理

python 03_onnx2rt.py



