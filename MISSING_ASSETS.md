# 复现所需的外部资源

## 必需数据

运行这份复现版本前，至少要准备：

- `train.csv`
- `train/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`

默认脚本假设它们位于：

- `/data/zilizhu/PE/train.csv`
- `/data/zilizhu/PE/train/`

如果你的实际位置不同，运行前设置：

```bash
export PE_TRAIN_CSV=/your/path/train.csv
export PE_TRAIN_DIR=/your/path/train
```

## 仓库内已包含的资源

- `trainval/lung_localization/lung_bbox.csv`
- `trainall/lung_localization/lung_bbox.csv`
- `pretrainedmodels/`
- `efficientnet_pytorch/`
- `albumentations/`

这些都已经复制进 `repro/`，不需要你再手动搬一次。

## 可能仍然需要的外部下载

训练时如果本机缓存里没有权重，下面这些 ImageNet 预训练权重会尝试自动下载：

- SE-ResNeXt50 / SE-ResNeXt101
- EfficientNet-B0

如果运行机器不能联网，需要提前准备好这些预训练权重缓存。

## Python 环境

原始 README 里给出的依赖版本是：

- torch 1.3.1
- torchvision 0.4.2
- pydicom 2.0.0
- pandas 1.0.3
- numpy 1.17.2
- gdcm 2.8.9
- apex 0.1
- transformers 2.11.0
- imgaug

如果你当前环境版本更高，也许仍然能跑，但不保证与冠军原始环境完全一致。

## 已知运行前检查项

- `python -m torch.distributed.launch` 在你环境中可用
- `apex` 已安装
- `train.csv` 字段与 Kaggle 原始数据一致
- DICOM 目录层级是 `Study/Series/*.dcm`

## 原始代码中保留的阶段依赖

- 肺定位阶段会生成 `bbox_dict_train.pickle` 和 `bbox_dict_valid.pickle`
- 图像级阶段会生成 `features0/*.npy`
- 二级模型会读取这些中间产物继续训练

所以第一次跑必须按 `run_trainval.sh` 或 `run_trainall.sh` 的完整顺序执行，不能跳阶段。

## 这份复现版本没有额外补的内容

- 没有新增 Kaggle test 推理脚本
- 没有替换原始损失函数或模型结构
- 没有引入你自己的双分支结构

这份目录的目标只是先把冠军方案本体跑通。
