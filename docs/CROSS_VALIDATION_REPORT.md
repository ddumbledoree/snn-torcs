# SNN-TORCS 移植依赖核查报告

> 核查日期: 2026-10-14（本地 Windows，只读核查 + CPU 推理冒烟测试，未部署）
> 范围: `snn_torcs_handoff/`（交接推理包）+ `acceptance_exp012_H20_20260824/`（验收证据包）
> 目标: 确认能否从零在远程 Linux (Docker) 上跑起来，并给出 GitHub 仓库化前必须修复的清单。

---

## 0. 结论摘要

| 项 | 状态 | 说明 |
|---|---|---|
| 交接包 `run.sh` 验收路径（demo + 回放 JSON） | ✅ 可跑 | Python 侧依赖自洽，模型/代码/配置交叉验证全部一致 |
| 验收包 metric1（时延基准） | ⚠️ 1 处断链 | `benchmark_policy_latency.py` 导入缺失的 `snn_torcs/trainer.py`，一行可修 |
| 验收包 metric2（H20 标准条件） | ✅ 布局修复后可跑 | 依赖仓库根布局重建 |
| 验收包 7 条件矩阵 | ❌ 缺模块 | 依赖 `snn_torcs/condition.py`，两个包里都没有，无法从本包恢复 |
| TORCS 仿真器本体 | ❌ 不在包内 | 必须在远端重建（vtorcs-RL-color 系，UDP 3101），是最大的部署工作量 |
| 硬编码路径 | ⚠️ 6 处 | `/home/desc/...`，多数有 CLI 覆盖，2 类需要改脚本 |
| GPU 依赖 | ⚠️ | 3 个验收 .sh 硬编码 `--device cuda[:0]`，CPU docker 直接崩 |

**总体判定**：`run.sh` 的接口验收闭环（交接包单独拎出来）在"TORCS 已装好 + conda 环境建好"的前提下可以直接跑通；验收包的复现脚本需要先重建仓库根布局并修 1 个 import、补 1 个缺失模块；真正"从零"跑起来的瓶颈不在本包代码，而在 TORCS 仿真器的重建（两个包都不含 TORCS 本体，README 也明确"贵方自备"）。

---

## 1. 两个包的角色

```
snn_torcs_handoff/                  ← 推理运行包（自洽，对应原仓库 /home/desc/gym_torcs/snn_torcs_handoff/）
  run.sh                            一键验收入口
  environment.yml                   conda 环境（python 3.13 + torch 2.12 + gym 0.26.2 + numpy 2.4.6）
  app/                              运行目录 = 原仓库根的推理子集
  model/exp013_best.pth             baseline SNN T=8 示例模型
  data/replay_example.json          预生成逐帧回放（1000 步）
  reports/                          交接时证据

acceptance_exp012_H20_20260824/     ← 验收证据包（原仓库放在 outputs/ 下）
  scripts/                          原仓库 scripts/ 的拷贝（含 run_*.sh 复现入口）
  models/                           exp012_ours_t4 + exp013_baseline_t8
  conditions/conditions.yaml        条件预设（湿滑/换车/换赛道）
  configs/default.yaml              原仓库 configs 拷贝
  reports/ runtime_notes/           验收证据与运行日志
```

原环境（由日志/报告元数据证实）：仓库在 `/home/desc/gym_torcs/`，conda 在 `/home/desc/Conda/envs/torcs_RL`，GPU `cuda:0`，TORCS UDP 端口 **3101**。

---

## 2. 已验证一致的项（交叉验证通过）

### 2.1 权重 ↔ 代码 ↔ 文档 三方一致
- `snn_torcs_handoff/model/exp013_best.pth` 与 `acceptance_.../models/exp013_baseline_t8_best.pth` **sha256 完全相同**：`22825d7b…`，且与 acceptance/README.md 声明值一致。
- `exp012_ours_t4_best.pth` sha256 `61c9982f…` 与 README 声明一致；其 checkpoint 元数据（episode=3665, best_reward=35601.36, stage=ours_t4）与 `reports/standard/baseline_t0.15_steptrace.json` 中的记录逐字段吻合。

### 2.2 checkpoint 结构 ↔ models.py 结构一致（本机 CPU 实测）
两个 .pth 均为 trainer 全量 checkpoint（含 `policy_net` 包装 + optimizer/buffer/rng_state），`snn_torcs/io.py::load_state_dict` 正确解包；state_dict 形状 `fc1(256,68) fc2(256,256) fc3(7,256)` 与 `FCSNN(68,256,7)` 完全匹配。本机 torch 实测：
- T=8 加载 exp013 → forward → Q(1,7) ✓
- T=4 加载 exp012 → forward → Q(1,7) ✓
- `weights_only=False` 已显式传入（torch≥2.6 默认翻转时不会炸）✓

### 2.3 交接包 Python import 闭包完整
`run.sh → app/eval_steptrace.py → snn_torcs{models,io,env_wrapper,utils} → gym_torcs → snakeoil3_gym`，全部文件在包内，无外部仓库依赖。`snn_torcs/run_mode.py` 仅用标准库。

### 2.4 数据自洽
`data/replay_example.json` 可解析，runs[0] 1000 步，字段与 eval_steptrace.py 输出 schema 一致，config 段记录其由 snn_ours + fast_train 生成。

### 2.5 environment.yml 与原环境一致
runtime_notes 日志证实原环境就是 python3.13/numpy2.4/gym0.26.2 组合（gym 打印 deprecation 警告但仍正常工作），不是"未验证的愿望清单"。

---

## 3. 断链与缺失项（必须处理）

### 🔴 A1. `snn_torcs/condition.py` 缺失 —— 7 条件矩阵无法复现
- 位置：`acceptance_.../scripts/eval_condition.py:22` → `from snn_torcs import condition as cond_mod`
- 两个包内的 `snn_torcs/` 都只有 `__init__/env_wrapper/io/models/run_mode/snn_core/utils` 七个文件，**没有 condition.py**。
- 该模块承载：conditions.yaml 加载、TORCS 赛道/车辆/摩擦的 symlink-farm 零写入切换、污染门（assert_stock_unchanged）、信号恢复。无法从包内其他文件重建。
- 影响：`run_7_condition_h20_matrix.sh` 全部 8 个条件都跑不了。
- 处置：从原仓库 `/home/desc/gym_torcs/snn_torcs/condition.py` 取回；取不到则把矩阵脚本标记为"证据存档、不可复现"，metric1/metric2 仍可跑。

### 🔴 A2. `snn_torcs/trainer.py` 缺失 —— metric1 断链（一行可修）
- 位置：`acceptance_.../scripts/benchmark_policy_latency.py:21` → `from snn_torcs.trainer import load_state_dict`
- io.py 文档串已说明"trainer.py re-exports these"，交接时抽离成了 io.py，但 benchmark 脚本没同步改。
- 修复：改为 `from snn_torcs.io import load_state_dict`（语义相同，io.py 就是抽出来的那份）。

### 🔴 A3. 仓库根布局假设 —— 三个 run_*.sh 全部依赖
- 每个 .sh 都有 `REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"` 然后 `cd "$REPO_ROOT"` 调 `scripts/xxx.py`，并依赖 cwd 下的 `configs/default.yaml`。
- 原布局：验收包在 `<repo>/outputs/acceptance_exp012_H20_20260824/`，repo 根还有 `scripts/、snn_torcs/、gym_torcs.py、snakeoil3_gym.py、autostart.sh、configs/、conditions/`。
- 现状：两个文件夹平铺上传 GitHub 的话，REPO_ROOT 会指到仓库根（两文件夹的上级），那里没有 scripts/ → 全挂。
- 修复方案见 §6 建议布局（把验收包放回 `outputs/` 下 + 用 app/ 的文件补齐 repo 根）。

### 🔴 A4. TORCS 本体不在包内（最大工作量）
代码通过 4 个界面依赖外部 TORCS：
1. `gym_torcs.py:244` `subprocess.Popen(['torcs', '-nofuel', '-nodamage', '-nolaptime'])` —— PATH 里要有 `torcs`
2. `snakeoil3_gym.py` Client 连 **UDP 3101**（`gym_torcs.py:180` `Client(p=3101)`）—— 标准 TORCS 是 3001，3101 是 vtorcs-RL 系改版的默认；即需要 **SCR server 补丁版 TORCS**（README 风险提醒里点名的 `vtorcs-RL-color`，即 `ugo-nama-kun/gym_torcs` 配套版本，基于 vtorcs-nosegfault）
3. `run_mode.py:26-28` 读写 `~/.torcs/config/raceman/practice.xml` 和 `~/.torcs/config/screen.xml` —— TORCS 至少要成功跑过一次生成用户配置；practice.xml 还须指向 g-track-1(road) + scr_server 驱动
4. `eval_condition.py`（经缺失的 condition.py）还要改写赛道 XML（湿滑摩擦、e-track-1、car2-trb1）→ 需要 **g-track-1、e-track-1 两条赛道 + car1-trb1、car2-trb1 两辆车**（标准 TORCS 数据包自带这些 track/car）

### 🟡 B1. GPU 硬编码
- `run_metric1_latency.sh:17` `--device cuda`；`run_metric2_standard_h20.sh:12` `--device cuda:0`；`run_7_condition_h20_matrix.sh:9` `--device cuda:0`。
- 远端 docker 无 NVIDIA GPU 时直接崩。`PY` 变量可环境覆盖，device 不行。建议改成 `DEVICE="${DEVICE:-cuda}"` 透传。
- 若保留 GPU：environment.yml 的 CUDA13 wheel 栈（torch 2.12 + cuda-toolkit 13 + 全套 nvidia-*，数 GB）必须装，且 docker 要有 NVIDIA Container Toolkit。

### 🟡 B2. `/home/desc` 硬编码（6 处）
| 文件 | 内容 | 风险 |
|---|---|---|
| 两个 `configs/default.yaml:62-65` | paths.data/ckpt/log/report_dir | 低：入口脚本都显式传了 `--report`；仅缺省回退时踩到。建议改相对路径 |
| 三个 `run_*.sh:6` | `PY=/home/desc/Conda/envs/torcs_RL/bin/python` | 低：`PY=xxx bash run_xxx.sh` 可覆盖，但要在 README 写明 |

### 🟡 B3. cwd 依赖的 `autostart.sh`
`gym_torcs.py:284` `Popen(['bash','autostart.sh'])` 与 `snakeoil3_gym.py` 兜底 `os.system('sh autostart.sh')` 都是**相对当前目录**。交接包布局 run.sh 先 `cd app/` ✓；仓库根布局（metric2/矩阵从 repo root 起跑）要求 repo 根也有一份 autostart.sh。两处都要保证。

### 🟡 B4. 两版 eval_steptrace.py 有分叉
- 验收包版多了 `--fixed-accel` 覆盖与 `done/done_reason` 字段；run_metric2/矩阵脚本用了 `--fixed-accel` → **repo 根 scripts/ 必须放验收包的新版**。
- 交接包 app/ 里是旧版（run.sh 不用 --fixed-accel，仍能跑）。建议同步成新版，消除分叉。

---

## 4. 外部依赖总清单（远端 docker 需要准备的东西）

### apt 包
```
xvfb  x11-utils(xdpyinfo)  xautomation(xte)  procps(pkill)  bash
# TORCS 源码编译（若选源码装）还需:
libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libplib-dev
libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev
```

### TORCS
- SCR 补丁版（vtorcs-RL-color / torcs-1.3.7+scr-server 一类），scr_server 驱动、UDP 3101
- 自带数据里须含 tracks: `g-track-1`(road), `e-track-1`(road)；cars: `car1-trb1`, `car2-trb1`
- 首次运行生成 `~/.torcs/`，且 `practice.xml` 配为 g-track-1 + scr_server
- 老代码库，新 GCC 可能要小补丁 —— 建议镜像基底选 Ubuntu 20.04/22.04

### conda
- `conda env create -f environment.yml && conda activate torcs_RL`（prefix 行会被忽略，无害）

---

## 5. 建议 GitHub 仓库布局（重建后满足全部路径假设）

```
snn-torcs/                          ← 仓库根 = 原 /home/desc/gym_torcs 的最小重建
├── run.sh                          （来自 snn_torcs_handoff/run.sh，或改为调 handoff/run.sh）
├── autostart.sh                    （来自 app/autostart.sh —— repo 根需要一份，见 B3）
├── gym_torcs.py                    （来自 app/gym_torcs.py）
├── snakeoil3_gym.py                （来自 app/snakeoil3_gym.py）
├── snn_torcs/                      （来自 app/snn_torcs/，7 个文件）
├── configs/default.yaml            （用验收包版或 app 版均可，建议同步并把 paths 改相对）
├── conditions/conditions.yaml      （来自验收包 —— 供补回的 condition.py 使用）
├── scripts/                        （来自验收包 scripts/ 的 4 个 .py + 3 个 .sh）
├── environment.yml
├── snn_torcs_handoff/              （原样保留，run.sh 自洽子包）
│   └── model/exp013_best.pth
└── outputs/
    └── acceptance_exp012_H20_20260824/   （原样保留 —— 保证 REPO_ROOT=PKG_DIR/../.. 成立）
        └── models/{exp012_ours_t4,exp013_baseline_t8}_best.pth
```

关键点：
- 验收包必须位于 `outputs/` 之下（A3 的 `../..` 才指向仓库根）。
- repo 根的 `scripts/eval_steptrace.py` 用验收包新版（B4）。
- `benchmark_policy_latency.py` 改一行 import（A2）。
- condition.py 补回后放 `snn_torcs/condition.py`（A1）。

## 6. 最小修复清单（按优先级）

**P0（不改就跑不起来）**
1. 按 §5 重建布局（纯文件移动，无代码改动）
2. `scripts/benchmark_policy_latency.py:21` `from snn_torcs.trainer import` → `from snn_torcs.io import`
3. 远端安装 TORCS + apt 工具链 + conda env（§4）
4. 从原仓库取回 `snn_torcs/condition.py`（否则矩阵脚本降级为不可复现）

**P1（健壮性）**
5. 三个 run_*.sh：`PY` 默认改 `command -v python || conda run -n torcs_RL python`；`--device` 改 `DEVICE` 环境变量透传，默认 cuda、CPU 机可 `DEVICE=cpu` 覆盖
6. 两个 configs/default.yaml 的 paths.* 改 `./outputs/...` 相对路径
7. app/eval_steptrace.py 同步验收包新版（--fixed-accel + done 字段）

**P2（仓库卫生，Windows→GitHub→Linux）**
8. 加 `.gitattributes`：`* text=auto eol=lf`（防 Windows 提交把 .sh/.py 转 CRLF——当前文件全是 LF，别让 git 破坏）
9. `git update-index --chmod=+x` 确保 run.sh/autostart.sh/run_*.sh 带执行位；README 同时写"用 `bash run.sh` 兜底"
10. 两个 .pth 共 ~85MB，低于 GitHub 100MB 单文件上限，可直传；想更干净可上 Git LFS
11. `rerun/` 是空目录，git 不收录（无害，脚本有 `mkdir -p`）

## 7. 远端部署验收 gate（给部署 agent 的检查点）

按顺序，每步通过才进下一步：
1. `conda activate torcs_RL && python -c "import torch,yaml,numpy,gym"` ✓
2. `command -v torcs Xvfb xdpyinfo xte pkill` 全部存在 ✓
3. `torcs` 首跑生成 `~/.torcs/config/raceman/practice.xml`，确认 track=g-track-1(road)、driver 含 scr_server、端口 3101 ✓
4. 纯推理冒烟（不依赖 TORCS）：`python scripts/benchmark_policy_latency.py --baseline-ckpt ... --ours-ckpt ... --device cpu`（改完 A2 后应打印两组时延 + reduction）✓
5. 交接包闭环：`bash snn_torcs_handoff/run.sh` → 看到 per-step 打印 + `TORCS terminated.` + `data/replay_example.json` 更新 ✓ —— **到这一步即"从零跑起来"达成**
6. （可选）metric2 H20；再（可选）补 condition.py 后跑 7 条件矩阵

---

### 附：本次核查方法备忘
- 全量 import 扫描（grep ^import/^from + 惰性 import 人工核对）
- 全量硬编码路径扫描（/home、/usr、/opt）
- sha256 三方比对（两包权重 ↔ README 声明）
- 本机 torch CPU 加载 + forward 冒烟（PlumeTracing 环境，KMP_DUPLICATE_LIB_OK=TRUE）
- 日志/报告元数据反推原环境（cuda:0、3101、/home/desc/gym_torcs）
- web 检索确认 vtorcs-RL-color = ugo-nama-kun/gym_torcs 配套 TORCS（UDP 3101 默认）
