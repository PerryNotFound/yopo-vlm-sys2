# yopo-vlm-sys2

[English](README.md) | [中文](README_zh.md)

YOPO-Rally 的 **System-2 语义安全层**：YOPO（System-1）基于深度图做几何规划（10Hz），
无法识别水面、沙地、泥泞等**深度相机看不见的语义危险**；本模块用 Qwen3-VL-4B
作为低频语义系统，通过**候选轨迹离散否决（Semantic Veto）**为规划器补上语义感知。

## 架构

```
                ┌─────────────────────────────────────────────┐
 RGB 图像 ─────►│ 颜色分割定位（帧级，快） + VLM 语义确认（慢）  │
 /odom ────────►│                    ↓                        │
 /visualize/* ─►│ 危险区域 → 轨迹-区域匹配 → /vlm_veto 否决掩码 │
                │ 全否决 → RECOVERY → /vlm_goal 扇区引导        │
                └─────────────────────────────────────────────┘
                                      ↓
   yopo_node：被否决候选代价置 +inf → argmin 只在安全候选中选择
   （树木等几何障碍仍由 YOPO 深度代价处理，两个系统是交集关系）
```

设计原则：异步解耦（不阻塞 yopo 10Hz / mpc 20Hz）、软超时回退（2s 无更新回退纯 YOPO）、
单点失败不瘫痪（VLM 推理失败静默跳过）。

## 实验定义（M0–M4）

| 实验 | 名称 | 方法 | 代码 | 状态 |
|---|---|---|---|---|
| **M0** | Baseline | 纯 YOPO + MPC，无语义感知 | 只跑 `diablo_algorithm.launch.py` | ✅ |
| **M1** | Frame-level Veto | VLM 帧级二值判断，危险即截停 `/cmd_vel` | `eval/m1_safety.py` + `eval/m1_arbitrator.py` | ✅ |
| **M2** | Candidate Semantic Veto | 危险区域 → 否决穿过该区域的候选轨迹（硬否决） | `eval/m2_veto.py` | ✅ |
| **M3** | Positive Fusion | 语义安全分数融入代价函数（软融合对照） | 待写 | ⬜ |
| **M4** | Veto + Recovery | 全否决时的回退策略（扇区引导 / 后退 / 重规划） | `eval/m2_veto.py` 的 RECOVERY 雏形 → 待独立 | 🔶 |

M2 感知消融（`m2_veto.py` 顶部 `MODE` 切换）：

| 模式 | 说明 | 论文角色 |
|---|---|---|
| `color` | 纯 HSV 颜色分割定位 | 基线：无法区分蓝色水面与蓝色物体 |
| `vlm_zone` | 纯 VLM 区域判断 | 基线：4B 模型空间定位不可靠（左右混淆、抖动） |
| `hybrid` | 颜色定位 + VLM 二值语义确认 | 本文方法 |

## 目录结构

```
yopo-vlm-sys2/
├── eval/               # 闭环实验节点（M0–M4）与离线评测
│   ├── m1_safety.py        # M1: VLM 帧级安全判断
│   ├── m1_arbitrator.py    # M1: cmd_vel 仲裁（转发/截停）
│   ├── m2_veto.py          # M2+M4雏形: 候选轨迹语义否决 + 恢复状态机
│   ├── eval_offline.py     # 离线准确率评测（基座 vs LoRA）
│   └── eval_debug.py       # 抽样打印 VLM 原始输出（调 prompt 用）
├── data/               # 语义数据集构建（900 帧，地面/水面/沙地/泥地 4 类）
├── finetune/           # QLoRA 微调（Qwen3-VL-4B，train/val JSONL + 训练脚本）
├── deploy/             # 部署工具（test_semantic.py 单图冒烟测试）
├── archive/            # 开发历程中的探索原型（附各自失效原因，见 archive/README.md）
└── docs/               # 工作手册与改动说明
```

## 运行

```bash
# 1. 终端A：ROS-TCP-Endpoint
source /opt/ros/humble/setup.bash
source ~/YOPO-Rally-Whole/tcp_ws/install/setup.bash
ros2 launch ros_tcp_endpoint endpoint.py

# 2. Unity 打开 YOPO-Sim-Project，加载 EvaluationScene，点 Play、Connect

# 3. 终端B：YOPO + MPC（M0 到此为止）
source /opt/ros/humble/setup.bash
cd ~/YOPO-Rally-Whole/YOPO-Rally && source install/setup.bash
ros2 launch diablo_ros_wrapper diablo_algorithm.launch.py

# 4. 终端C：VLM 节点（M1 / M2 按需）
conda activate base
source ~/YOPO-Rally-Whole/YOPO-Rally/.venv/bin/activate
python eval/m2_veto.py        # M2；M1 则跑 m1_safety.py + m1_arbitrator.py

# 5. RViz 选目标点（2D Goal Pose）
```

> 注意：`yopo_node` 的 `/vlm_veto` 接口改动后必须重新编译：
> `colcon build --packages-select diablo_ros_wrapper`（禁止 `--symlink-install`）

## 评测指标（论文用）

通过 `ros2 bag` 录制 `/vlm_state`、`/vlm_veto`、`/odom` 等话题统计：
成功率、VLM 干预次数、误干预率、距危险地形最近距离、到达目标用时、受控停车（STUCK）次数。
录制脚本见 `YOPO-Rally/scripts/record_all.sh`。

## 模型

- 基座：`Qwen3-VL-4B-Instruct`（4bit NF4 量化，单次推理约 0.2–0.3s）
- 微调：QLoRA（r=16），数据 900 帧。当前微调版（纯色块数据）val 准确率 53.9%
  低于基座 82.6%，闭环实验均使用基座模型；纹理增强数据 v2 在路线图上。
