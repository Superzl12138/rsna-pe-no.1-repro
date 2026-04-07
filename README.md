# RSNA PE 冠军方案复现

这个目录整理出了一份可直接运行的冠军方案复现版本，保留了原始流水线结构，并把执行入口统一到了 `repro/` 下。

## 目录

- `trainval/`：本地验证流程
- `trainall/`：全量训练流程
- `setup_data_links.sh`：把当前机器上的数据路径映射到原始代码期望的 `input/` 结构
- `run_trainval.sh`：前台执行本地验证复现
- `run_trainall.sh`：前台执行全量训练复现
- `start_trainval_bg.sh`：后台启动本地验证并写总日志
- `start_trainall_bg.sh`：后台启动全量训练并写总日志
- `MISSING_ASSETS.md`：缺失数据、预训练权重和环境要求说明

## 数据路径

原始代码固定读取：

- `../../input/train.csv`
- `../../input/train/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`

这里通过软链接兼容原始路径约定。默认会把下面两个实际路径映射进来：

- `PE_DATA_ROOT=/data/zilizhu/PE`
- `PE_TRAIN_CSV=$PE_DATA_ROOT/train.csv`
- `PE_TRAIN_DIR=$PE_DATA_ROOT/train`

如果你的真实路径不同，直接在运行前覆盖环境变量即可。

## 快速开始

### 本地验证

```bash
cd /path/to/repro
bash run_trainval.sh
```

### 本地验证后台运行

```bash
cd /path/to/repro
bash start_trainval_bg.sh
```

### 全量训练

```bash
cd /path/to/repro
bash run_trainall.sh
```

### 全量训练后台运行

```bash
cd /path/to/repro
bash start_trainall_bg.sh
```

## 常用环境变量

### 改数据路径

```bash
export PE_DATA_ROOT=/data/zilizhu/PE
export PE_TRAIN_CSV=/data/zilizhu/PE/train.csv
export PE_TRAIN_DIR=/data/zilizhu/PE/train
```

### 改多卡数量

```bash
export NPROC_PER_NODE=4
```

### 改二级模型所用 GPU

```bash
export SECOND_LEVEL_GPU=0
```

### 改总日志目录（默认：`/data/zilizhu/PE/repro_outputs`）

```bash
export LOG_DIR=/path/to/repro/logs
```

## 已处理的兼容项

- 保留了原始相对路径读取逻辑，避免大规模改代码
- 修正了 `trainval/seresnext101/run.sh` 中错误的特征导出脚本名
- 把多卡数量从硬编码 `4` 改成了 `NPROC_PER_NODE`
- 把二级模型的 `CUDA_VISIBLE_DEVICES=0` 改成了 `SECOND_LEVEL_GPU`
- 新增后台启动脚本，统一把顶层输出写入 `logs/` 目录

## 运行顺序

### trainval

1. `process_input`
2. `lung_localization/split2`
3. `seresnext50`
4. `seresnext101`
5. `2nd_level`

### trainall

1. `process_input`
2. `lung_localization/splitall`
3. `seresnext50`
4. `seresnext101`
5. `2nd_level`

## 建议

先跑 `trainval`，确认数据、依赖和预训练下载都没问题，再跑 `trainall`。

## 日志

- 各阶段原始日志仍然会写入各自子目录下的 `*.txt`
- 后台启动脚本会额外生成一个总日志（默认写入 `/data/zilizhu/PE/repro_outputs`）：
  - `trainval_时间戳.log`
  - `trainall_时间戳.log`

## 复用既有 pickle（跳过 process_input）

如果你已经生成过 `series_dict.pickle/image_dict.pickle/…` 等文件（例如位于 `/home/zilizhu/CAD_PE-main/process_input/split3`），可以直接复用，避免重新扫描 DICOM：

```bash
cd /path/to/repro
export SKIP_PROCESS_INPUT=1
export EXISTING_PICKLE_DIR=/home/zilizhu/CAD_PE-main/process_input/split3
bash start_trainval_bg.sh
```

脚本会把上述目录中的必需文件以软链接的形式放到 `trainval/process_input/split2/`，后续阶段将直接读取。
