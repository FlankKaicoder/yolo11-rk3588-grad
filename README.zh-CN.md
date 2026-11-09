# yolo11-rk3588-grad 项目改动与算法尝试记录

本文档用于记录当前仓库相对原始 Ultralytics YOLO 工程所加入的工程改造、数据处理流程、算法改进尝试和实验脚本。当前整理基于仓库内代码静态阅读，重点覆盖自定义目录、训练入口、模型配置、蒸馏模块、patch 分类实验、数据构建脚本和导出脚本。

> 说明：仓库中仍保留大量 Ultralytics 上游代码、文档、示例和测试文件；本文只记录本项目实际新增或明显围绕毕业设计任务改造的部分。

## 项目目标

本项目围绕工业缺陷/孔洞类视觉任务开展，主要目标包括：

- 基于 YOLO11 做缺陷检测与实例分割训练。
- 面向小目标缺陷增强 YOLO 检测头，例如加入 P2/4 小目标检测分支。
- 尝试 ECA、CBAM、SimAM 等轻量注意力模块，改善细粒度缺陷区域表达。
- 从分割标注中构造 patch 分类数据，训练 ResNet18 类别表征模型。
- 针对类别不均衡和细粒度混淆，尝试 CE、SupCon、BCL、PaCo、DCL、cRT + Balanced Softmax 等表征学习方案。
- 将 patch 分类 teacher 的局部语义知识蒸馏到 YOLO11 分割 student 中。
- 支持蒸馏模型剥离 teacher/adapter 后导出纯 YOLO student，便于后续部署。
- 面向 RK3588/RKNN 部署保留 Ultralytics 导出能力和工程结构。

## 自定义代码地图

| 路径                                                            | 作用                                                                      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `train.py`                                                      | YOLO11n-seg 分割 baseline 训练入口，使用旧分割数据集。                    |
| `train_3clsbaseline.py`                                         | YOLO11n-seg 三分类分割 baseline 训练入口。                                |
| `train_yolo_seg_distill.py`                                     | 第一版正样本 ROI 分割蒸馏训练入口。                                       |
| `train_yolo_seg_fgbg_distill.py`                                | 第二版前景/背景 ROI 分割蒸馏训练入口。                                    |
| `run_seg_distill.py`                                            | 第一版蒸馏训练命令封装。                                                  |
| `inspect_seg_layers.py`                                         | 注册 forward hook，查看 YOLO11-seg 各层输出 shape，用于选择蒸馏 hook 层。 |
| `export_student_only_seg.py`                                    | 从蒸馏 checkpoint 中过滤出纯 student 权重。                               |
| `distill_seg_v1(positive_ROI)/`                                 | 第一版只使用 GT 正样本 ROI 的分割蒸馏模块。                               |
| `distill_seg_v2_fgbg/`                                          | 第二版前景/近背景 ROI 双分支蒸馏模块。                                    |
| `scripts/dataset/`                                              | 数据清洗、类别合并、单类转换、二分类 patch 数据构造脚本。                 |
| `scripts/cls/train_patch_binary_ce.py`                          | defect/background 二分类 patch teacher 训练脚本。                         |
| `scripts/train_patch_*.py`                                      | 多类 patch 分类、对比学习、长尾学习实验脚本。                             |
| `scripts/patch_resnet18_*.py`                                   | cRT、Balanced Softmax、DCL zoom-positive 等额外表征学习尝试。             |
| `scripts/analyze_repr_50_models.py`                             | 对不同 patch 模型抽特征，做线性探测、KNN、类中心距离、t-SNE 分析。        |
| `scripts/extract_misclassified_from_report.py`                  | 从分类报告中抽取误分类样本并按 true->pred 分类保存。                      |
| `scripts/extract_pairwise_focus.py`                             | 汇总重点类别对的类中心距离。                                              |
| `ultralytics/cfg/models/11/yolo11_p2*.yaml`                     | YOLO11 检测结构改造配置，加入 P2 分支和注意力模块。                       |
| `ultralytics/nn/modules/block.py`                               | 新增 ECAAttention、CBAM、SimAM 等模块实现。                               |
| `ultralytics/nn/tasks.py`、`ultralytics/nn/modules/__init__.py` | 让 YAML 中的新模块可被模型解析和构建。                                    |
| `my_configs/`                                                   | 检测实验批量训练 shell 配置。                                             |

## 数据处理与数据集改造

### 1. 清洗分割标签中的重复 bbox

代码：`scripts/dataset/clean_new_dataseg_remove_bbox_duplicates.py`

目的：处理同一个 label 文件中同时存在 YOLO-seg polygon 和疑似 5 列 bbox 的情况。

主要逻辑：

- 读取 `new_dataseg` 的 `train/val/test`。
- 对每个 label 行判断格式：
  - polygon：`class x1 y1 x2 y2 ...`，列数大于 5 且为奇数。
  - 5 列 bbox：`class x y w h` 或类似 bbox 行。
- 保留 polygon。
- 如果 5 列 bbox 与同类 polygon 的外接框高度重合，则认为是重复框并删除。
- 如果 5 列 bbox 无法和 polygon 对齐，则写入 `suspicious_5col_*.txt`，方便人工复查。
- 输出 clean 数据集和新的 `data.yaml`。

意义：保证分割训练只使用 polygon 标注，避免 bbox 行混入分割标签导致训练语义混乱。

### 2. 多类分割转单类 defect

代码：`scripts/dataset/make_seg_1cls_defect_keep_split.py`

目的：把原多类别缺陷统一为一个 `defect` 类，用于“先检测/分割缺陷，再细分类型”的层级方案。

主要逻辑：

- 复制原始图片目录结构。
- 读取原 YOLO-seg polygon label。
- 跳过明显的 detect bbox 五列标签。
- 将所有 polygon 的 class id 改成 `0`。
- 保留 train/val/test 划分。
- 输出 `nc: 1`、`names: ["defect"]` 的 `data.yaml`。

意义：降低分割阶段类别难度，让 YOLO 先专注于“哪里有缺陷”。

### 3. 四分类分割转三分类分割

代码：`scripts/dataset/make_seg_3classes_from_clean_polygon.py`

目的：把细粒度中相近的 `missing_coating` 和 `missing_material` 合并，缓解类别边界不稳定。

类别映射：

| 原类别                | 新类别         |
| --------------------- | -------------- |
| `0: missing_coating`  | `0: missing`   |
| `2: missing_material` | `0: missing`   |
| `1: corrosion`        | `1: corrosion` |
| `3: carbon`           | `2: carbon`    |

意义：用三分类验证“合并易混类别是否能提升稳定性”。

### 4. 检测数据转单类 defect

代码：`scripts/dataset/make_detect_1cls_defect.py`

目的：把原检测数据的所有类别统一为 `defect`，用于单类检测 baseline 或层级两阶段方案。

主要逻辑：

- 复制 `images/train|val|test`。
- 将所有 label 的第一列 class id 改为 `0`。
- 对没有 label 的图片创建空 txt，保留负样本。
- 输出单类检测 `data.yaml`。

### 5. 从分割数据构造二分类 patch 数据

代码：`scripts/dataset/build_patch_binary_v1.py`

目的：从 YOLO-seg 数据集中裁剪 patch，构造 `defect` vs `background` 二分类数据，用来训练二分类 teacher。

生成的 patch 类型：

- `defect`：基于 GT polygon 的外接框裁剪，并按 `expand_ratio` 扩张。
- `easy_bg`：在整图中随机采背景框，要求与任意 GT 的 IoU 不超过阈值。
- `near_bg`：在每个 GT 的上下左右邻近区域采困难背景，要求与 GT 的 IoU 足够小。

关键参数：

- `patch-size`：默认 224。
- `expand-ratio`：默认 0.15。
- `easy-bg-per-image`：每张图随机背景 patch 数。
- `near-bg-per-defect`：每个缺陷附近背景 patch 数。
- `max-iou-bg-with-gt`：背景框和 GT 的最大允许 IoU，默认 0.05。

输出：

- ImageFolder 风格目录：`train/defect`、`train/background`、`val/defect`、`val/background`。
- `meta/*.csv`：记录 patch 来源、坐标、类型、IoU。
- `stats/class_distribution.csv` 和 `stats/dataset_summary.txt`。

意义：为后续 defect/background teacher 和前景/背景蒸馏提供数据基础。

## YOLO 检测结构改造

### 1. P2 小目标检测头

代码：`ultralytics/cfg/models/11/yolo11_p2.yaml`

原 YOLO11 检测头通常输出 P3/8、P4/16、P5/32 三个尺度。本项目新增 P2/4 分支：

```text
backbone P2 -> head P2/4 -> Detect(P2, P3, P4, P5)
```

核心改动：

- 在 head 中继续上采样到 P2/4。
- 与 backbone 第 2 层的 P2 特征 concat。
- 新增 `C3k2 [128]` 生成 P2 小目标特征。
- Detect 输入从 `[P3, P4, P5]` 扩展为 `[P2, P3, P4, P5]`。

意义：提升小缺陷、小孔洞、细小异常区域的召回能力。

### 2. ECA 注意力

代码：

- `ultralytics/nn/modules/block.py`
- `ultralytics/cfg/models/11/yolo11_eca.yaml`
- `ultralytics/cfg/models/11/yolo11_p2_eca.yaml`

实现：`ECAAttention`

核心逻辑：

- 对特征做全局平均池化。
- 使用 1D 卷积建模局部通道交互。
- 通过 sigmoid 生成通道权重。
- 用通道权重重新标定输入特征。

实验配置：

- `yolo11_eca.yaml`：在常规 P3/P4/P5 检测结构中加入 ECA。
- `yolo11_p2_eca.yaml`：在 P2 小目标分支后加入 ECA。

意义：以很小参数量增强通道选择能力，尝试改善细粒度缺陷纹理表达。

### 3. CBAM 注意力

代码：

- `ultralytics/nn/modules/block.py`
- `ultralytics/cfg/models/11/yolo11_p2_cbam.yaml`

实现模块：

- `ChannelAttention`
- `SpatialAttention`
- `CBAM`

核心逻辑：

- 先做通道注意力：avg pool + max pool + MLP。
- 再做空间注意力：通道维 avg/max 拼接后用卷积生成空间权重。
- 最终同时强化重要通道和重要空间位置。

意义：验证通道+空间联合注意力对缺陷检测的帮助。

### 4. SimAM 注意力

代码：

- `ultralytics/nn/modules/block.py`
- `ultralytics/cfg/models/11/yolo11_p2_simam.yaml`

实现：`SimAM`

核心逻辑：

- 使用无参数能量函数估计每个神经元的重要性。
- 不额外引入卷积/线性层。
- 输出 `x * sigmoid(energy)`。

意义：验证无参数注意力在小目标缺陷场景下的收益。

### 5. 解析器接入

代码：

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

改动：

- 将 `ECAAttention`、`CBAM`、`SimAM` 暴露给模块注册表。
- 在 `parse_model` 中为这些模块补齐输入通道 `c1`，并设置输出通道 `c2 = c1`。

没有这一步，YAML 中写 `ECAAttention`、`CBAM`、`SimAM` 会无法构建模型。

### 6. 检测实验 shell

代码：

- `my_configs/train_hole4cls_3exp.sh`
- `my_configs/train_hole4cls_simam_cbam.sh`
- `my_configs/hole_detect_dataset.yaml`

实验组合：

- YOLO11n baseline。
- YOLO11n + P2。
- YOLO11n + P2 + ECA。
- YOLO11n + P2 + SimAM。
- YOLO11n + P2 + CBAM。

训练设置：

- `epochs=200`
- `imgsz=640`
- `batch=16`
- `seed=42`
- `pretrained=yolo11n.pt` 用于自定义结构初始化。

## Patch 分类与表征学习实验

这部分主要服务两个目标：

1. 直接研究缺陷小 patch 的细粒度分类能力。
2. 训练后续蒸馏用的 teacher 特征模型。

### 1. ResNet18 + CE baseline

代码：

- `scripts/train_patch_baseline.py`
- `scripts/train_patch_ce_fix_v1.py`
- `scripts/train_patch_ce_ep150_v2.py`

演进：

- 初始版 `train_patch_baseline.py` 使用 ImageFolder、ResNet18 ImageNet 预训练、CE loss、WeightedRandomSampler。
- `train_patch_ce_fix_v1.py` 修正为更干净的控制实验：
  - 固定随机种子。
  - 使用 ImageNet mean/std normalize。
  - 不使用 sampler。
  - 不使用 class-balanced CE。
  - 保存 `best_acc.pth`、`best_macro_f1.pth`、`last.pth`。
  - 输出 `metrics.csv`、`train.log`、每轮 classification report。
- `train_patch_ce_ep150_v2.py` 将同一设置扩展到 150 epochs，用于和更长训练的 BCL 对齐。

意义：作为所有对比学习/长尾学习方法的公平基线。

### 2. SupCon 系列

代码：

- `scripts/train_patch_supcon.py`
- `scripts/train_patch_supcon_light.py`
- `scripts/train_patch_supcon_heave.py`
- `scripts/train_supcon_twostage.py`
- `scripts/train_yolo_supcon.py`

共同思想：

- 每个样本生成两个增强视角。
- ResNet18 或 YOLO11 分类 backbone 输出投影特征。
- 使用 supervised contrastive loss 拉近同类样本，推远异类样本。
- 同时保留 CE 分类头。

单阶段版本：

- `train_patch_supcon.py`：`LAMBDA_CE=0.7`，`LAMBDA_SUPCON=0.3`。
- `train_patch_supcon_light.py`：降低 SupCon 权重，`LAMBDA_SUPCON=0.1`。
- `train_patch_supcon_heave.py`：提高 SupCon 权重，`LAMBDA_SUPCON=0.4`。

两阶段版本：

- `train_supcon_twostage.py`
  - Stage 1：SupCon 表征预训练。
  - Stage 2：分类头训练/微调。
  - 使用较大 batch、SGD、CosineAnnealingLR。

YOLO 分类 backbone 版本：

- `train_yolo_supcon.py`
  - 将 YOLO11 分类模型包装为 SupCon encoder。
  - Stage 1 训练投影头。
  - Stage 2 使用 CE 做分类微调。

意义：验证监督对比学习能否改善细粒度缺陷类别之间的表征间隔。

### 3. BCL 系列

代码：

- `scripts/train_patch_bcl.py`
- `scripts/train_patch_bcl_fix_v1.py`
- `scripts/train_patch_bcl_ep150_v2.py`
- `scripts/train_patch_bcl_2stage_fix_v1.py`
- `scripts/train_patch_bcl_2stage_freeze_ep150_v2.py`
- `scripts/train_patch_bcl_2stage_freeze_ep50_v3.py`

模型结构：`BCLResNet18`

- ResNet18 encoder。
- Projection head：`Linear(512)->ReLU->Linear(128)`。
- Classifier head：`Linear(512)->num_classes`。
- forward 返回 `feat, proj, logits`。

初始 BCL：

- `train_patch_bcl.py`
- 使用 CE + Balanced Contrastive Loss。
- 同时尝试 class-balanced CE 和 WeightedRandomSampler。

修正版 BCL：

- `train_patch_bcl_fix_v1.py`
- 关键修复：修正 contrastive features 展开顺序。
- 正确做法是：

```python
contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
```

这样顺序为：

```text
[view1 全部样本, view2 全部样本]
```

与 `mask.repeat(n_views, n_views)` 对齐。

控制变量：

- 去掉 sampler。
- 去掉 class-balanced CE。
- 保持与 CE baseline 相同的数据增强、normalize、优化器和记录方式。

150 epoch 版本：

- `train_patch_bcl_ep150_v2.py`
- 保持修正版 BCL 设置，延长到 150 epochs。

两阶段直接微调：

- `train_patch_bcl_2stage_fix_v1.py`
- 总共 50 epochs：
  - Stage 1：25 epochs BCL pretrain。
  - Stage 2：25 epochs CE finetune。

两阶段冻结微调：

- `train_patch_bcl_2stage_freeze_ep150_v2.py`
- 总共 150 epochs：
  - Stage 1：60 epochs BCL pretrain。
  - Phase 1：10 epochs 只训练 classifier。
  - Phase 2：20 epochs 解冻 layer4 + classifier。
  - Phase 3：60 epochs 全网低学习率微调。

- `train_patch_bcl_2stage_freeze_ep50_v3.py`
- 总共 50 epochs：
  - Stage 1：20 epochs BCL pretrain。
  - Phase 1：5 epochs classifier-only。
  - Phase 2：10 epochs layer4 + classifier。
  - Phase 3：15 epochs 全网低学习率微调。

意义：系统比较“单阶段 CE+BCL”和“先表征预训练再分类微调”的差异，并研究冻结策略是否能保护对比学习得到的类间结构。

### 4. PaCo proxy contrastive

代码：`scripts/train_patch_paco.py`

主要尝试：

- 使用 `PaCoProxyLoss`。
- 为每个类别维护 proxy。
- 样本特征与类别 proxy 做温度缩放相似度。
- 与 CE 组合：

```text
loss = LAMBDA_CE * ce_loss + LAMBDA_PACO * paco_loss
```

其他设置：

- 使用 class-balanced weights。
- 使用 WeightedRandomSampler。

意义：验证 proxy-based contrastive 在类别不均衡 patch 分类中的效果。

### 5. DCL + zoom-positive 两阶段

代码：`scripts/patch_resnet18_dcl_zoom_twostage.py`

主要尝试：

- 构造 weak view 和 zoom view。
- zoom view 类似“局部更紧裁剪”的正样本视角。
- 使用 `BalancedBatchSampler` 保证每个 batch 内类别更均衡。
- Stage 1：Supervised DCL pretrain。
- Stage 2：冻结 encoder，训练 linear classifier。
- Stage 3：解冻 ResNet18 layer4 + classifier 做 CE 微调。

意义：尝试让模型对同一缺陷的局部缩放视角保持一致，并增加类间分离。

### 6. cRT + Balanced Softmax

代码：`scripts/patch_resnet18_crt_balsoftmax.py`

主要尝试：

- Stage 1：自然采样下用 CE 训练完整 ResNet18，学习表征。
- Stage 2：冻结 backbone，重新初始化 classifier。
- 使用 `BalancedSoftmaxLoss` 重新训练分类头。

意义：针对长尾类别不均衡，验证 classifier re-training 是否能减少头部类别偏置。

### 7. defect/background 二分类 teacher

代码：`scripts/cls/train_patch_binary_ce.py`

用途：训练前景/背景蒸馏使用的二分类 teacher。

特点：

- 输入数据为 `build_patch_binary_v1.py` 生成的 ImageFolder 数据。
- 模型为 ResNet18。
- 支持 ImageNet 预训练。
- 使用 CE loss。
- 使用 AdamW + CosineAnnealingLR。
- 以 macro F1 做 early stopping 和 best checkpoint 选择。
- 输出 confusion matrix、loss 曲线、accuracy/F1 曲线、history CSV、metrics JSON。

意义：teacher 不直接服务最终推理，而是给分割 student 提供 defect/background patch 级语义特征。

## Patch 表征分析与错误分析

### 1. 多模型表征对比

代码：`scripts/analyze_repr_50_models.py`

比较对象：

- CE 50 epoch。
- BCL 50 epoch。
- BCL two-stage direct。
- BCL two-stage freeze。

分析内容：

- 抽取 train/val 特征并保存为 `.npz`。
- 训练 Logistic Regression linear probe。
- 训练 KNN probe。
- 计算类内距离。
- 计算类中心距离矩阵。
- 绘制 val t-SNE。
- 汇总 `summary_all_models.csv`。

意义：不只看最终分类准确率，还观察表征空间是否真的让易混类别分开。

### 2. 重点类别对距离提取

代码：`scripts/extract_pairwise_focus.py`

关注类别对：

- `corrosion <-> missing_material`
- `corrosion <-> missing_coating`
- `missing_material <-> missing_coating`
- `carbon <-> corrosion`
- `carbon <-> missing_material`

意义：专门观察最容易混淆的缺陷类别之间的中心距离变化。

### 3. 误分类样本整理

代码：`scripts/extract_misclassified_from_report.py`

功能：

- 从某轮 report JSON 中读取样本预测结果。
- 找出所有误分类样本。
- 输出 `misclassified.csv`。
- 按 `true_class__TO__pred_class` 创建目录并复制/软链接图片。
- 输出误分类 pair 统计。

意义：辅助人工查看“模型到底把哪些缺陷搞混了”。

## YOLO 分割训练与数据路线

### 1. 分割 baseline

代码：

- `train.py`
- `train_3clsbaseline.py`

实验：

- `train.py`：YOLO11n-seg 在旧分割数据上训练 baseline。
- `train_3clsbaseline.py`：YOLO11n-seg 在三分类分割数据上训练 baseline。

共同设置：

- `epochs=200`
- `imgsz=640`
- `batch=32`
- `optimizer="auto"`

### 2. layer shape 检查

代码：`inspect_seg_layers.py`

用途：

- 给 YOLO11-seg 每层注册 forward hook。
- 输入 dummy image。
- 打印每层输出 shape。
- 用于选择蒸馏 hook 层。

当前蒸馏默认使用 `hook_idx=13`。对 `yolo11n-seg` 来说，该层是 head 中 P4/16 融合特征，通道数通常为 128，因此默认 `student_feat_dim=128`。

## 分割蒸馏 v1：正样本 ROI 蒸馏

代码目录：`distill_seg_v1(positive_ROI)/`

训练入口：`train_yolo_seg_distill.py`

核心思想：

- student 是 YOLO11-seg。
- teacher 是 patch 分类模型 `BCLTeacher`。
- 只使用 GT 缺陷框作为正样本 ROI。
- teacher 看原图 crop。
- student 看 YOLO hook 层对应 ROI pooled feature。
- 使用 cosine loss 对齐 student/teacher 特征。

数据流：

```text
原图 + GT box
  |-- crop GT patch -> BCLTeacher -> teacher feature [1, 512]
  |-- YOLO hook feature -> ROI pool -> adapter -> student feature [1, 512]
  |-- cosine_distill_loss(student, teacher)
```

总损失：

```text
total_loss = seg_loss + lambda_dist * dist_loss * batch_size
```

模块职责：

| 文件                 | 作用                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| `model.py`           | 包装 YOLO11-seg，注册 hook，重写 criterion，合并原始分割 loss 与蒸馏 loss。                          |
| `teacher_wrapper.py` | 定义 `BCLResNet18` 和 `BCLTeacher`，加载 patch BCL checkpoint，返回 ResNet18 avgpool 后 512 维特征。 |
| `adapter.py`         | 将 student ROI 特征映射到 teacher 特征维度。                                                         |
| `roi_pool.py`        | 原图 crop 与特征图 ROI pooling。                                                                     |
| `losses.py`          | cosine 蒸馏损失。                                                                                    |

意义：先验证“patch teacher 的缺陷语义能否迁移到 YOLO 分割中间层”。

## 分割蒸馏 v2：前景/背景 ROI 双分支蒸馏

代码目录：`distill_seg_v2_fgbg/`

训练入口：`train_yolo_seg_fgbg_distill.py`

这是当前更完整的蒸馏方案。

### 核心变化

v1 只对缺陷 GT 框做正样本蒸馏。v2 引入了背景分支：

- `pos_boxes`：GT 缺陷框。
- `neg_boxes`：缺陷附近、但与 GT IoU 很小的 near background 框。

总损失：

```text
dist_loss = lambda_pos * pos_loss + lambda_neg * neg_loss
total_loss = seg_loss + dist_loss * batch_size
```

### teacher

代码：`distill_seg_v2_fgbg/teacher_wrapper.py`

teacher 为二分类 ResNet18：

- 加载 `model_state_dict`。
- 根据 `fc.weight` 推断类别数。
- forward 不返回分类 logits，而返回 `fc` 前的 512 维特征。
- teacher 参数冻结，并使用 `torch.no_grad()`。

处理流程：

```text
crop -> clamp(0,1) -> resize 224x224 -> ImageNet normalize -> ResNet18 -> feat [B,512]
```

### student adapter

代码：`distill_seg_v2_fgbg/adapter.py`

结构：

```text
Flatten -> Linear(in_dim, 512) -> ReLU -> Linear(512, 512)
```

默认：

- `in_dim=128`
- `out_dim=512`

作用：把 YOLO hook 层 ROI pooled feature 映射到 teacher 的 512 维空间。

### ROI 采样

代码：`distill_seg_v2_fgbg/fgbg_sampler.py`

前景：

- 对 batch 中每个 GT box，转成像素级 `xyxy`。
- 直接加入 `pos_boxes`。

近背景：

- 对每个 GT，在 `left/right/top/bottom` 四个方向随机采样。
- 背景框宽高大约为 GT 的 `0.5 ~ 0.8` 倍，且不小于 `min_crop_size`。
- 候选框必须满足：

```text
max_iou_with_gt(candidate, all_gt_boxes) <= max_iou_bg_with_gt
```

默认 `max_iou_bg_with_gt=0.05`。

意义：背景分支不是随机容易背景，而是“靠近缺陷、容易混淆、但不覆盖 GT 的困难背景”。

### ROI crop 与 ROI pooling

代码：`distill_seg_v2_fgbg/roi_pool.py`

teacher 路径：

```text
imgs[bi] -> crop_image_tensor(box_xyxy) -> crop
```

student 路径：

```text
hook feature[bi] -> box 从原图坐标映射到特征图坐标 -> adaptive_avg_pool2d -> roi_feat
```

如果 crop 或 ROI 无效，则跳过该样本。

### 单分支损失

代码：`distill_seg_v2_fgbg/model.py` 中 `_compute_one_branch_loss`

对传入的一组 box 重复执行：

```text
同一个 ROI:
  原图 crop -> teacher -> t [1,512]
  特征图 ROI -> adapter -> s [1,512]
  收集 t/s

所有有效 ROI:
  cat -> [N,512]
  cosine_distill_loss(student_feats, teacher_feats)
```

注意：这个函数本身不区分前景/背景。前景或背景由外部传入的 `sampled_boxes` 决定。

### cosine 蒸馏损失

代码：`distill_seg_v2_fgbg/losses.py`

公式：

```text
student = normalize(student)
teacher = normalize(teacher)
loss = 1 - mean(sum(student * teacher))
```

含义：

- 只约束特征方向，不强制特征模长。
- student 与 teacher 越相似，loss 越接近 0。

### hook 与 checkpoint 保存

代码：`distill_seg_v2_fgbg/model.py`

模型初始化时：

- 根据 `hook_idx` 找到 YOLO module。
- 注册 forward hook。
- 每次 forward 保存中间层输出到 `_distill_feat`。

保存 checkpoint 前：

- `DistillSegTrainer.save_model()` 会移除 raw model 和 EMA model 里的 hook。
- 保存后重新给 raw model 注册 hook。

意义：避免 hook 被 deepcopy/EMA/checkpoint 序列化后重复或残留。

### 当前 v2 注意事项

- `easy_bg_per_image` 已作为参数传入，但 `distill_seg_v2_fgbg/fgbg_sampler.py` 当前实际没有生成 easy background；只生成 near background。
- `student_feat_dim=128` 与 `yolo11n-seg + hook_idx=13` 匹配。更换模型尺寸或 hook 层时必须重新确认通道数。
- teacher checkpoint 格式要求包含 `model_state_dict`，且需要有 `fc.weight`。

## 蒸馏模型 student-only 导出

代码：

- `export_student_only_seg.py`
- `scripts/extract_student_only_from_distill_seg.py`

目的：蒸馏训练 checkpoint 内包含 teacher、adapter、hook 等训练期组件，部署时只需要 YOLO student。

主要逻辑：

- 加载 distill checkpoint。
- 优先取 EMA model。
- 构建干净的 YOLO student。
- 过滤掉 `teacher.*`、`adapter.*` 等蒸馏专用参数。
- 只加载 shape 匹配的 student 参数。
- 复制 `names`、`nc` 等类别信息。
- 保存为可直接 `YOLO(out_path)` 加载的 student-only checkpoint。

意义：让蒸馏收益落到最终 YOLO 分割模型上，同时不把 teacher 带到推理端。

## RK3588 / RKNN 相关

仓库名称中包含 `rk3588`，当前工程保留了 Ultralytics 的 RKNN 导出路径：

- `ultralytics/engine/exporter.py` 支持 `format=rknn`。
- RKNN 默认目标平台为 `rk3588`。
- `ultralytics/nn/autobackend.py` 包含 RKNN 推理后端检测。

与本项目直接相关的部署前置步骤是：

1. 蒸馏训练得到 full distill checkpoint。
2. 使用 student-only 导出脚本剥离训练期 teacher/adapter。
3. 对纯 YOLO student 执行 ONNX/RKNN 等部署导出。

## 当前实验路线总览

可以把整个项目理解为四条互相连接的路线：

```text
路线 A：YOLO 检测结构改造
baseline -> P2 -> P2+ECA / P2+CBAM / P2+SimAM

路线 B：分割数据与 baseline
raw seg -> clean polygon -> 1cls/3cls seg -> YOLO11n-seg baseline

路线 C：patch 表征学习
seg/detect data -> patch dataset -> CE/SupCon/BCL/PaCo/DCL/cRT teacher

路线 D：分割蒸馏
patch teacher + YOLO11-seg student
  -> v1 positive ROI distill
  -> v2 foreground/background ROI distill
  -> student-only export
```

## 主要算法尝试清单

| 方向           | 尝试                    | 代码位置                                                                         | 目的                                     |
| -------------- | ----------------------- | -------------------------------------------------------------------------------- | ---------------------------------------- |
| 小目标检测     | P2/4 检测分支           | `yolo11_p2.yaml`                                                                 | 提升小缺陷召回。                         |
| 注意力         | ECA                     | `block.py`、`yolo11_eca.yaml`、`yolo11_p2_eca.yaml`                              | 加强通道选择。                           |
| 注意力         | CBAM                    | `block.py`、`yolo11_p2_cbam.yaml`                                                | 同时建模通道和空间注意力。               |
| 注意力         | SimAM                   | `block.py`、`yolo11_p2_simam.yaml`                                               | 无参数注意力，低成本增强。               |
| 类别重组       | 4 类转 3 类             | `make_seg_3classes_from_clean_polygon.py`                                        | 合并易混 missing 类。                    |
| 层级检测       | 多类转单类 defect       | `make_seg_1cls_defect_keep_split.py`、`make_detect_1cls_defect.py`               | 先定位缺陷，再做细分。                   |
| patch 数据     | defect/background patch | `build_patch_binary_v1.py`                                                       | 训练二分类 teacher。                     |
| patch baseline | ResNet18 + CE           | `train_patch_ce_fix_v1.py`                                                       | 建立公平基线。                           |
| 表征学习       | SupCon                  | `train_patch_supcon*.py`                                                         | 增大类间距离。                           |
| 表征学习       | YOLO SupCon             | `train_yolo_supcon.py`                                                           | 验证 YOLO 分类 backbone 做 patch 表征。  |
| 表征学习       | BCL                     | `train_patch_bcl*.py`                                                            | 类别均衡监督对比学习。                   |
| 表征学习       | BCL two-stage           | `train_patch_bcl_2stage*.py`                                                     | 先学表征，再微调分类。                   |
| 长尾分类       | PaCo                    | `train_patch_paco.py`                                                            | proxy contrastive 缓解类别不均衡。       |
| 长尾分类       | cRT + Balanced Softmax  | `patch_resnet18_crt_balsoftmax.py`                                               | 冻结表征，重训分类头。                   |
| 对比学习       | DCL + zoom-positive     | `patch_resnet18_dcl_zoom_twostage.py`                                            | 让局部缩放视角保持一致。                 |
| 分割蒸馏       | positive ROI distill    | `distill_seg_v1(positive_ROI)/`                                                  | 把 patch teacher 的缺陷表征蒸馏到 YOLO。 |
| 分割蒸馏       | FG/BG ROI distill       | `distill_seg_v2_fgbg/`                                                           | 同时约束缺陷区域和近背景区域。           |
| 部署准备       | student-only export     | `export_student_only_seg.py`、`scripts/extract_student_only_from_distill_seg.py` | 剥离 teacher/adapter，保留推理模型。     |

## 复现实验建议顺序

建议按下面顺序复现，比较不容易乱：

1. 清洗分割标签：

```bash
python scripts/dataset/clean_new_dataseg_remove_bbox_duplicates.py
```

2. 生成单类或三类分割数据：

```bash
python scripts/dataset/make_seg_1cls_defect_keep_split.py --src-yaml <clean_data.yaml> --dst-root <out_root>
python scripts/dataset/make_seg_3classes_from_clean_polygon.py
```

3. 训练 YOLO11-seg baseline：

```bash
python train.py
python train_3clsbaseline.py
```

4. 构造二分类 patch 数据：

```bash
python scripts/dataset/build_patch_binary_v1.py --src-root <seg_dataset> --dst-root <patch_binary_dataset>
```

5. 训练二分类 teacher：

```bash
python scripts/cls/train_patch_binary_ce.py --data-root <patch_binary_dataset> --save-dir <save_dir>
```

6. 训练 FG/BG 蒸馏分割模型：

```bash
python train_yolo_seg_fgbg_distill.py \
  --seg-model yolo11n-seg.pt \
  --data \
  \
  13 \
  --lambda-pos 0.1 \
  --lambda-neg 0.1 < seg_data.yaml > --teacher-ckpt < binary_teacher_best_macro_f1.pth > --hook-idx
```

7. 导出 student-only：

```bash
python export_student_only_seg.py \
  --distill-ckpt <distill_best.pt> \
  --base-model yolo11n-seg.pt \
  --out <best_student_only.pt>
```

8. 再对 student-only 模型做验证或部署导出。

## 当前代码注意事项

- 很多脚本写死了 `/root/autodl-tmp/yolo11-rk3588-grad/...` 路径；换机器或 Windows 本地运行前需要改路径。
- `distill_seg_v1(positive_ROI)` 是历史目录名，但其中 import 写的是 `distill_seg.*`；如果直接运行第一版蒸馏，需要确认包名是否已经在实际训练环境中做过重命名或软链接。
- `distill_seg_v2_fgbg` 的 `easy_bg_per_image` 参数目前没有在训练时采样器中真正使用；真正启用的是 near background。
- `hook_idx` 与 `student_feat_dim` 强绑定。默认 `hook_idx=13`、`student_feat_dim=128` 适配 `yolo11n-seg`，换模型尺寸或换层需要重新用 `inspect_seg_layers.py` 确认。
- patch 多类实验和二分类 teacher 使用的 checkpoint 格式不同；蒸馏 teacher wrapper 也不同，不能混用。
- 历史文件中部分中文注释出现乱码，但代码结构和变量名仍能表达主要意图；本 README 已按 UTF-8 中文重新整理。
- 当前 README 记录的是“尝试过的代码方案”，不等价于最终实验结论；最终结论应结合 `runs/` 下指标、混淆矩阵、表征分析结果再写入论文。

##

可以将本项目的改进概括为三个层次：

1. 数据层：清洗 polygon 标注，构造单类/三类分割数据，生成 defect/background patch 数据。
2. 模型层：YOLO11 加 P2 小目标检测头，引入 ECA/CBAM/SimAM 注意力模块。
3. 训练策略层：使用 patch 级表征学习训练 teacher，并通过前景/背景 ROI 蒸馏将 teacher 的局部语义迁移到 YOLO11 分割 student。

最核心的创新尝试是：

```text
二分类 patch teacher + YOLO11-seg student + foreground/background ROI feature distillation
```

它不是直接蒸馏 teacher 的分类概率，而是让 YOLO 中间层在缺陷 ROI 和近背景 ROI 上分别对齐 teacher 的 patch 级语义特征，从而同时强化缺陷表达和背景抑制能力。
