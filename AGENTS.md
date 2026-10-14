# AGENTS.md — 部署 Agent 操作说明

## 你的任务

在这台无头 Linux（docker）上从零部署 TORCS + 本仓库，最终**复现两个验收指标**：

1. 指标① 决策速度：`latency_reduction ≥ 0.10`（归档值 0.4258）
2. 指标② H20 解码控制有效率：`decode_precision ≥ 0.90`（归档值 0.9050）

成功判据（G3）：`bash run.sh` 打印 per-step 数据 + `TORCS terminated.` +
`snn_torcs_handoff/data/replay_example.json` 重新生成。

## 必读

- `DEPLOY_TORCS.md` —— 完整部署步骤（apt 依赖 → 编译 vtorcs-RL-color → practice.xml → conda env → G1-G4 gate）
- `README.md` —— 仓库结构与指标口径
- `docs/CROSS_VALIDATION_REPORT.md` —— 移植核查背景（哪些文件从哪来、为何这样摆）

## 关键事实（不要重新发现）

- **TORCS 变体必须是 vtorcs-RL-color**（github.com/ugo-nama-kun/gym_torcs）：
  `snakeoil3_gym.py` 连 **UDP 3101**（标准 TORCS 是 3001，装错了会永远 `Waiting for server`）。
- 代码自动管理显示：fast_train 模式下自己起 `Xvfb :99`、自己 `xte` 按键进 Practice（autostart.sh）、
  自己备份/改写/恢复 `~/.torcs` 的 practice.xml 和 screen.xml。你只需保证 `torcs` 在 PATH、
  practice.xml 初始指向 g-track-1 + scr_server（模板 `docs/practice.g-track-1.xml`）。
- `~/.torcs` 只有 TORCS 首次启动后才存在；先用 `timeout 15 xvfb-run -a torcs` 生成再装模板。
- 验收脚本：`PY=<python路径>` 覆盖解释器；`DEVICE=cpu` 覆盖算力（无 GPU 机器必用）。
- 指标①（benchmark_policy_latency.py）**不需要 TORCS**，可最先跑，用来验证 Python 侧全链路。
- 鲁棒性矩阵（run_7_condition_h20_matrix.sh）依赖 `snn_torcs/condition.py`，
  本仓库是**占位 stub**（重建规格见 docs/condition_module_spec.md）。主 gate ①② 不需要它，
  不要因为矩阵跑不了而阻塞主验收。

## 红线

- **不改** `outputs/acceptance_exp012_H20_20260824/`（归档证据）和 `snn_torcs_handoff/`（冻结交接包）
  的任何已有文件；新证据只写 `outputs/.../rerun/`（已 gitignore）。
- **不改** `.gitattributes`（LF 强制，Windows 协作者依赖它）。
- 遇到要改代码的情况：先最小改动 + 在 PR/commit 里说明理由；涉及验收口径（H20 定义、
  指标计算、模型结构）的文件（models/snn_core/summarize 脚本）**一个字都不要动**。
- alt-track 条件的 CommTimeout（~556 步）是已知仿真器问题，不要试图"修"它，也不得把
  pre-timeout 数据当完整 1000 步结果上报。

## 工作方式

- 按 DEPLOY_TORCS.md 的 G1→G4 逐 gate 推进，每 gate 留下输出证据（终端日志贴到汇报里）。
- 编译 TORCS 报错时：优先 `./configure --disable-xrandr`、降并行度、或换 ubuntu:20.04 基底，
  而不是大改上游源码。
- 每完成一个 gate 做一次 git commit（信息格式：`gate-G<n>: <结果摘要>`）。
- 全部 gate 通过后：汇报 metric1/metric2 的 json 数值 vs 归档值（README 表格）。
