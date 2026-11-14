# M5：纯 VLM 语义避障方法说明

> 代码：`yopo-vlm-sys2/eval/vlm_semantic_avoid.py`（节点名 `vlm_semantic_avoid`）
> 仿真配套：`YOPO-Sim-Project/Assets/Scripts/PuddleRandomizer.cs` + `WaterDepthBased.mat`
> 撰写日期：2026-11-05，状态：闭环验证通过（VLM 实时报出 `hazards=[('water','center')]` 并正确否决候选轨迹）

---

## 1. 方法定位与动机

M5 是 M0–M4 消融系列之后的**主力方法**：底层运动规划完全交给 YOPO（深度几何避障 + 5 条候选轨迹 + MPC 跟踪），语义安全完全交给 VLM——**不再使用任何颜色分割/手工视觉特征**。

| 层 | 负责模块 | 输入 | 输出 |
|---|---|---|---|
| 几何层（底层） | YOPO + MPC（原系统，未改动） | 深度图、里程计 | 5 条候选轨迹 `/visualize/path_{0..4}`、速度指令 |
| 语义层（本方法） | `vlm_semantic_avoid.py` | RGB 图、候选轨迹、里程计 | `/vlm_veto` 否决掩码、`/vlm_goal` 恢复引导 |

**为什么彻底去掉颜色分割（M2 的 hybrid/color 模式）**：2026-11-04 的 HSV 标定实验（`deploy/sample_hsv.py`，K-means 聚类车相机画面）证明，半米高相机以掠射角看水面时，水面反射的是树木和天空，聚类结果中**不存在任何蓝色簇**（全部 H∈13~69 黄绿系，与草地/树影混在一起）。颜色法在该工况下是原理性失效，不是阈值没调好。而 VLM 的语义理解能力天然区分"水面反光"和"草地"，因此感知 100% 由 VLM 承担，颜色代码只保留在 m2/m3/m4 里作论文消融对照。

## 2. 系统架构与数据流

```
                ┌───────────────────── YOPO（不改）─────────────────────┐
 深度图 ──────► │ yopo_node：5 条候选轨迹 + 代价评估 + MPC               │
 /odom ───────► │  订阅 /vlm_veto：被否决候选代价置 +inf                 │
                │  订阅 /vlm_goal：恢复态软引导候选扇区                  │
                └──────▲───────────────────────────▲──────────────────┘
                       │ /visualize/path_{0..4}     │ /vlm_veto /vlm_goal
                       │                            │
                ┌──────┴───────────────────────────┴──────────────────┐
 RGB ─────────► │ vlm_semantic_avoid（本方法）                         │
 /odom ───────► │  VLM 语义识别 → 分区危险 → 轨迹否决 → 恢复状态机     │
                │  /cmd_vel_mpc → /cmd_vel 转发（launch remap 约定）   │
                └────────────────────────────────────────────────────┘
```

### 话题接口

| 方向 | 话题 | 类型 | 用途 |
|---|---|---|---|
| 订阅 | `/camera/color/image/compressed` | CompressedImage | VLM 视觉输入 |
| 订阅 | `/odom` | Odometry | 机器人位姿（轨迹投影用） |
| 订阅 | `/visualize/path_{0..4}` | Path | YOPO 的 5 条候选轨迹 |
| 订阅 | `/cmd_vel_mpc` | Twist | MPC 输出，本节点转发 |
| 发布 | `/vlm_veto` | Int32MultiArray | 被否决候选的序号（yopo_node 置 +inf；2s 超时未更新自动回退纯 YOPO） |
| 发布 | `/vlm_goal` | PoseStamped | 恢复态引导目标（map 系，7m 外） |
| 发布 | `/vlm_state` | String | NORMAL / RECOVERY / STUCK，计指标用 |
| 发布 | `/vlm_hazards` | String | 防抖后生效的 hazards JSON（含类别），计指标用 |
| 发布 | `/vlm_veto_debug` | String | VLM 原始输出文本，调 prompt 用 |
| 发布 | `/cmd_vel` | Twist | 转发给仿真器 |

## 3. VLM 语义感知模块

### 3.1 模型与推理节奏

- 模型：Qwen3-VL-4B-Instruct（本地 `models/Qwen3-VL-4B-Instruct`），bitsandbytes 4bit NF4 量化加载；
- 推理线程独立于 ROS 回调：**只处理最新一帧**（`_last_infer_seq` 跳过积压帧），单帧延迟实测 0.2~0.45 s，等效约 2~5 Hz；推理失败静默跳过本轮，节点不瘫痪；
- `max_new_tokens=96`、greedy 解码（`do_sample=False`）保证输出格式稳定。

### 3.2 Prompt 设计（`vlm_semantic_avoid.py` 顶部 `PROMPT`）

设计要点：

1. **明确类别词表**：`water`（含浅水反光）/ `mud` / `sand` / `tree`，输出只能用这四个英文 token——省 token、易解析、避免中文同义词漂移；
2. **明确安全类**：草地、泥土路、落叶、树影、石头、天空——把最容易误报的（树影、天空蓝）提前排除；
3. **空间分区协议**：图像横向五等分 `far_left/left/center/right/far_right`，VLM 只需做粗粒度分区判断（4B 模型给像素级坐标不可靠，给五分区可靠得多）；
4. **边缘抑制**："只占区域边缘一小角或距离很远的目标不要报告"——抑制分区抖动和远处无关目标；
5. **严格 JSON 协议**：`{"hazards":[{"type":"water","zone":"center"}]}`，空则 `{"hazards":[]}`。

相比 M2 的"只输出危险区域列表"，hazards 协议让每次否决都能溯源到**语义类别**（`/vlm_hazards` 话题直接可记入实验指标），这是论文里"语义感知"的直接证据。

### 3.3 输出解析（`parse_hazards`）

三级解析，鲁棒优先：

1. 正则提取第一个 `{...}` JSON 块，`json.loads` 后取 `hazards` 列表；校验 `type ∈ CLASSES`、`zone ∈ ZONES`；
2. **兼容旧协议** `{"danger":["center"]}`（type 记 `unknown`），保证和 m2_veto 的调试数据互通；
3. JSON 解析失败的文本兜底：类别词与区域词按**字符距离贪心配对**（每处提及最多配一次；区域正则长词在前 `far_left|far_right|left|right|center`，防止 `left` 吃掉 `far_left`）。已知限制：纯中文倒装句（"center 有 water"）可能配错对——主路径在 greedy 解码下几乎总是合法 JSON，兜底极少触发。

### 3.4 类别过滤（`VETO_CLASSES`）

`vlm_semantic_avoid.py:66`：默认四类全参与否决。想让某类只识别不否决（例如把 `tree` 完全交给 YOPO 深度避障，避免和几何层重复决策），从 `VETO_CLASSES` 集合删除即可，`CLASSES`（识别词表）不动。

## 4. 几何映射：候选轨迹 → 图像分区

纯几何投影，不依赖相机内外参标定文件，只用一个水平视场角 `CAMERA_HFOV_DEG = 69.4°`：

1. **轨迹点变换到机体系**：`/odom` 取 (x, y, yaw)，轨迹点做平面刚体变换得 (local_x 前, local_y 左)；
2. **剔除**：`local_x ≤ 0.1`（身后/正侧方）跳过；`距离 > DANGER_RANGE_M = 10m`（距离门控）跳过；
3. **投影**：焦距 `fx = (img_w/2) / tan(hfov/2)`，方位角 `bearing = atan2(local_y, local_x)`，像素横坐标 `u = img_w/2 − fx·tan(bearing)`；
4. **FOV 外不否决**：`u` 超出图像宽度的点无法被视觉确认，保守起见不参与否决；
5. **分区**：`u` 均分为 5 段映射到五分区。每条轨迹均采 6 个点，**轨迹经过的区域集合 ∩ 被否决区域 ≠ ∅ → 该轨迹否决**。

## 5. 时间对称防抖

VLM 输出存在帧间抖动（同一画面相邻两帧可能报/不报），直接映射到否决集会导致行驶方向抽动。采用**时间制对称迟滞**（`_update_veto_state`）：

- 区域**进入**否决：危险需连续持续 `DANGER_CONFIRM_SEC = 0.25s`（VLM 单帧推理约 0.3s，即首次报出基本立即生效；曾为 0.5s，实测反应链过长导致车到水边才否决，已收紧）；
- 区域**退出**否决：安全需连续持续 `SAFE_RELEASE_SEC = 2.0s`。

实测日志里可以看到该机制的账面特征：VLM 已报 `hazards=[]` 后，`vetoed_zones=['center']` 仍会保持约 2 秒再解除——这是设计行为，不是 bug。

## 6. 否决语义与恢复状态机

主更新 10 Hz（`VETO_PUBLISH_PERIOD = 0.1s`，快于 VLM 推理，保证 yopo_node 的 2s 超时永不触发误回退）。

**安全余量扩展**（2026-11-05 加入）：地形类危险（`water/mud/sand`，`MARGIN_CLASSES`）的否决分区向左右各扩 `VETO_MARGIN_ZONES = 1` 个分区。动机：水坑跨分区交界时 VLM 往往只报一个区，贴分区边界的候选轨迹会"漏否决"擦进水里，表现为车在 RECOVERY 与 NORMAL 之间抖动、犹豫着蹭进水。扩展后 center 报水即否决 left/center/right 三个区，只留最外侧果断转向。`tree` 不扩——竖直障碍由 YOPO 深度层负责，扩了会在密林里过度保守。

- **NORMAL**：正常否决。只要 YOPO 扇区内还有未被否决的候选，车继续走；
- **RECOVERY**：5 条候选**全部被否决**持续 `RECOVERY_ENTER_SEC = 1s` 进入。每拍选一个逃生分区，**选取规则是"离危险最远"**（对每个安全分区计算它到最近否决分区的距离取最大，并列时取最靠中间的；优先保持上次选择防抖），朝该分区中心方位角发布 `/vlm_goal`（机器人前方 7m，map 系），把 YOPO 的候选扇区果断引离危险；视野内危险消失即退出，**退出时重置全否决计时器**——否则否决集 2s 释放迟滞未结束，`_all_veto_since` 旧值会让节点立即重新进入 RECOVERY，状态机在 NORMAL/RECOVERY 间以 10Hz 抖动（2026-11-05 实测发现并修复）；
- **STUCK**：恢复态下连安全分区都没有（四面环水），保持全否决停车等待——论文中计为**受控失败**，严格优于驶入危险。某方向解除后自动回到 RECOVERY。

## 7. 仿真侧配套：让 VLM"看得见"水（关键经验）

**VLM 识别率的上限不在模型，在渲染。** 本次调通前 VLM 连续报空（`hazards=[]` 刷满屏），根因全在仿真侧，最终生效配置如下。

### 7.1 水坑几何（`PuddleRandomizer.cs`）

| 参数 | 最终值 | 说明 |
|---|---|---|
| `surfaceOffset` | **0.08 m** | 水面 = 范围内**最低**原地面 + 8cm。抬高让水面离开草丛；太大会在水缘显得悬空 |
| `maxHeightDiff` | **0.05 m** | 选址平坦度上限。不挖坑后靠它保证贴地：地面起伏 ≤5cm，低处进水、高处自然露头形成岸线 |
| `maxAttemptsPerPuddle` | 100 | 平坦度收紧后选址变难，多试几次保证放满 6 个 |

**挖坑代码已整体移除（2026-11-05）**：地形完全不被修改（曾挖 0.4/0.15/0.08 m 三版洼地，问题分别是坑沿遮挡水面、坡度差、以及 Play 中断导致坑写进地形资产——最后一种曾真实发生，用 `git checkout -- Assets/TerrainData/EvaluationTerrainData.asset` 恢复）。现在水面就是铺在平地上的薄片，几何上对深度相机完全隐形，"可通行但语义危险"的实验前提更纯粹。

**⚠️ 关键教训：Inspector 序列化值优先于 C# 默认值**。组件挂进场景后改 `[SerializeField]` 默认值对已有场景实例无效——`maxHeightDiff` 在场景文件里一直是旧值 0.5 m，脚本里改 0.05 从未生效，这是"坡度差"问题久治不愈的真正根因。调参要在 Inspector 或场景文件里改。

### 7.2 水材质（`WaterDepthBased.mat`）

shader 是**深度差驱动**的（水面到水下地形的深度差决定透明度与浅/深水色插值），有两个直接后果：

- 水**形状无关紧要**（平面/圆柱混色结果一样），现在用的是代码生成的零厚度不规则 blob 面；
- 水太浅 → alpha≈0 → **水面透明到看不见**，VLM 无物可认。

| 参数 | 原值 | 最终值 | 作用 |
|---|---|---|---|
| `Shallow Water Color` | (0.16, 0.76, 0.98) 亮青 | **(0.10, 0.55, 0.85)** | 饱和蓝绿，低机位可辨 |
| `Deep Water Color` | (0.42, 0.47, 0.61) 灰蓝 | **(0.03, 0.18, 0.45)** | 原色几乎发灰，是"水发白"的主因之一 |
| `Smoothness` | 0.979 | **0.75** | 原值镜面过强，掠射角菲涅尔反光把天空白色全反射回来洗白水色 |
| `Depth Bias Alpha` / `Depth Contrast` | 0.5 / 0.5 | **0.95 / 5** | 不挖坑后水下深度仅 ~3cm，深度差驱动的 alpha 会塌成透明（水是"看不见"的），必须把透明度下限抬到 0.95 兜底；若仍嫌淡，终极手段是 Surface Type 改 Opaque |

## 8. 运行方法

```bash
# 终端 A：仿真（Unity Editor 点 Play，或跑编译好的 YOPO-Sim）
# 终端 B：算法栈（注意：这个终端不要 activate conda base，
#         conda 的 Python3.13 和 ROS Humble 的 Python3.10 冲突）
cd ~/YOPO-Rally-Whole/YOPO-Rally && source install/setup.bash
ros2 launch diablo_ros_wrapper diablo_algorithm.launch.py

# 终端 C：本节点（需要 conda base + venv 提供 torch/transformers）
conda activate base && source ~/YOPO-Rally-Whole/YOPO-Rally/.venv/bin/activate
cd ~/YOPO-Rally-Whole/yopo-vlm-sys2
python eval/vlm_semantic_avoid.py

# 终端 D（可选）：监视
ros2 topic echo /vlm_veto_debug   # VLM 原始输出，调 prompt 用
ros2 topic echo /vlm_hazards      # 防抖后生效的类别×区域
```

RViz 里用 2D Goal Pose 选目标点即可闭环。

**车速**：`diablo_ros_wrapper` 的 `yopo_node.py` 和 `mpc_node.py` 顶部 `vel_out_multiplier` 是全局速度缩放（两个文件必须一致，改完 `colcon build --packages-select diablo_ros_wrapper`）。当前 0.8（v_max = 1.6 m/s，角速度同步缩放，轨迹时间跨度拉长、空间形状不变）。历史：1.0（2 m/s）时反应链 1.2s ≈ 2.7m 反应距离，车到水边才否决；降到 0.6 后偏肉，0.8 是平衡点。

## 9. 实测现象（2026-11-05 闭环日志样例）

```
[INFO] [vlm_semantic_avoid]: VLM hazards=[('water', 'center')] latency=0.43s
[INFO] [vlm_semantic_avoid]: state=NORMAL danger=['center'] vetoed_zones=['center'] veto_paths=[0, 1, 2, 3, 4]
[INFO] [vlm_semantic_avoid]: VLM hazards=[('water', 'right')] latency=0.42s
[INFO] [vlm_semantic_avoid]: state=NORMAL danger=['right'] vetoed_zones=['center'] veto_paths=[1, 2, 3]
```

- 水进入画面中央约 0.5 s（防抖确认）后，中心分区被否决，穿过该区的候选轨迹全部被置 +inf；
- VLM 短暂报空后否决仍保持约 2 s（SAFE_RELEASE_SEC 迟滞），方向不抽动；
- 车随之向未否决侧绕行，闭环成立。

## 10. 已知限制与调参指引

| 现象 | 排查 |
|---|---|
| `hazards` 一直空 | 先 `ros2 topic echo /vlm_veto_debug` 看原始回答；再把相机截图和仿真画面对照——大概率是水/目标在图像里不可见（回第 7 节调渲染），其次才是 prompt |
| 区域左右混淆、抖动 | 4B 模型分区能力的固有噪声，已被 0.25s/2s 防抖吸收；恶化时可把 `DANGER_CONFIRM_SEC` 加回 0.5s，或微调（`finetune/` 里有 QLoRA 管线） |
| 误报树影/天空为 water | prompt 已显式排除；仍误报可在 `VETO_CLASSES` 里临时只留 `water` 调试，或加样本微调 |
| 恢复态绕不出去 | `/vlm_goal` 是软引导，YOPO 扇区受深度障碍约束；真被困会进 STUCK 停车，属设计行为（受控失败） |

## 11. 文件清单

| 文件 | 角色 |
|---|---|
| `yopo-vlm-sys2/eval/vlm_semantic_avoid.py` | 本方法全部代码（约 380 行，无外部依赖 beyond ROS/torch/transformers/cv2） |
| `YOPO-Rally/.../diablo_ros_wrapper/yopo_node.py` | YOPO 侧 VLM 接口消费方（`/vlm_veto` 置 +inf、`/vlm_goal` 软引导、2s 超时回退），未改动 |
| `YOPO-Sim-Project/Assets/Scripts/PuddleRandomizer.cs` | 随机水坑生成（不挖地、blob 水面、半径递减重试、Console 打印每个水坑坐标） |
| `YOPO-Sim-Project/Assets/Materials/VLMWater.mat` | **当前水材质**（2026-11-12 起）：URP/Lit 不透明亮蓝 `(0.10,0.40,0.88)`。透明深度材质在车相机里近黑被认成 mud，已弃用（WaterDepthBased.mat 保留可回切，见 YOPO-Sim使用指南 2.3.1） |
| `YOPO-Sim-Project/Assets/Samples/UnitySensorsROS/3.0.0/Sample/Prefabs/Camera/RGBCamera_ros.prefab` | RGB 相机 `_fov` 30°→70°（2026-11-12），VLM 视野预瞄水面命中率大幅提升 |
| `YOPO-Sim-Project/Assets/Editor/AutoPlay.cs` | 命令行 `-autoPlay` 启动：强制 Standalone 构建目标（WebGL 下 ROS-TCP 不通）+ 自动进 Play，全链路可脚本化 |
| `yopo-vlm-sys2/eval/m2_veto.py` 等 | M1–M4 消融对照，保留不动 |

### 2026-11-12 闭环复测

亮蓝水坑（r=1.7m）在 70° 视场下远距离即入画，北向穿越时 VLM 连续报 `('water','center')` 并触发 `veto_paths`；车绕行/通过后到达目标点。一次南向穿越未检出系水坑位于微洼地+草丛遮挡（物理不可见），非识别失败。手写 .meta 注意 GUID 必须 32 位十六进制，否则资产被静默忽略、渲染品红。
