# AGENTS.md — 部署 Agent 操作说明

## 你的任务

在这台无头 Linux（docker）上从零部署 TORCS + 本仓库，最终**复现两个验收指标**：
1. 指标① 决策速度：`latency_reduction ≥ 0.10`（归档值 0.4258）
2. 指标② H20 解码控制有效率：`decode_precision ≥ 0.90`（归档值 0.9050）

部署完成的判据（部署文档里的第 3 步检查点）：`bash run.sh` 打印 per-step 数据 + `TORCS terminated.` +
`data/replay_example.json` 重新生成。

## 必读

- `DEPLOY_TORCS.md` —— 完整部署步骤（apt 依赖 → 编译 vtorcs-RL-color → practice.xml → conda env → 四步检查清单）
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

- **不改** `outputs/acceptance_exp012_H20_20260824/`（归档证据）和 `docs/handoff_archive/`（交接归档）
  的任何已有文件；新证据只写 `outputs/.../rerun/`（已 gitignore）。
- **不改** `.gitattributes`（LF 强制，Windows 协作者依赖它）。
- 遇到要改代码的情况：先最小改动 + 在 PR/commit 里说明理由；涉及验收口径（H20 定义、
  指标计算、模型结构）的文件（models/snn_core/summarize 脚本）**一个字都不要动**。
- alt-track 条件的 CommTimeout（~556 步）是已知仿真器问题，不要试图"修"它，也不得把
  pre-timeout 数据当完整 1000 步结果上报。

## 已知部署目标（2026-10 平台容器快照，供参考，以实际检测为准）

- Ubuntu 22.04.1（与 DEPLOY_TORCS.md 推荐基底一致）、root 用户、x86_64、32C/64T
- NVIDIA RTX 4090 24GB，驱动 595.84（CUDA 13.2）→ environment.yml 的 CUDA13 栈可直接用，DEVICE=cuda 可用
- Anaconda 装在 /home/lynxi（非 root 的 $HOME）→ 验收脚本的 PY 自动探测已兼容 conda-on-PATH；
  兼容不了时手动 `PY=/home/lynxi/anaconda3/envs/torcs_RL/bin/python bash scripts/...`
- GitHub / PyPI 连通已验证；仓库若为 Private，clone 需 PAT 或 deploy key
- 持久化：/root 为容器 overlay，重建即丢。工作目录建议 /root/workspace/snn-torcs；
  TORCS 编译成功后立即打包备份：
  `tar czf /root/workspace/torcs-installed.tgz /usr/local/bin/torcs* /usr/local/share/games/torcs /usr/local/lib/torcs ~/.torcs`
  （重建容器后解包 + ldconfig 可免重新编译）
- GPU 可能与平台其他任务共享；eval 负载轻，影响小；时延基准想要干净数字可选空闲时段或 DEVICE=cpu 对照

## 工作方式

- 按 DEPLOY_TORCS.md 的四步检查清单（工具链 → 模型推理 → 一键自检 → 指标复现）逐步推进，每步留下输出证据（终端日志贴到汇报里）。
- 一次只跑一个评测进程（Xvfb :99 / UDP 3101 全局唯一，并行会互杀 torcs）。
- 指标判读用 gate 口径（①reduction≥0.10、②H20≥0.90），不要逐位对齐归档值；重新编译的
  TORCS 物理有微小差异，H20 落在 90% 附近小幅波动属正常。
- 编译 TORCS 报错时：优先 `./configure --disable-xrandr`、降并行度、或换 ubuntu:20.04 基底，
  而不是大改上游源码。
- 每完成一个 gate 做一次 git commit（信息格式：`gate-G<n>: <结果摘要>`）。
- 全部 gate 通过后：汇报 metric1/metric2 的 json 数值 vs 归档值（README 表格）。
