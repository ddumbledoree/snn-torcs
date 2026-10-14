# SNN-TORCS 推理与验收复现仓库

SNN（Spiking Neural Network）策略在 TORCS 上的**推理运行包 + 验收证据包**，
从原训练环境（`/home/desc/gym_torcs`，Linux + conda `torcs_RL` + GPU + vtorcs-RL-color）
移植为可独立部署的 GitHub 仓库。**本仓库不含训练代码**——只做"加载训好的模型 → 驱动 TORCS → 出验收指标"。

## 两条验收 gate（主指标）

| 指标 | 定义 | 归档实测 | 复现命令 |
|---|---|---|---|
| ① 决策速度 | ours(T=4) 前向时延 ≤ 0.9× baseline(T=8) | 42.58% 提升（pass） | `bash scripts/run_metric1_latency.sh` |
| ② H20 解码控制有效率（标准条件） | baseline_t0.15 / g-track-1 / 1000 步 | 90.50%（pass） | `bash scripts/run_metric2_standard_h20.sh` |

鲁棒性矩阵（wet/icy/换车/换赛道 6+1 条件，均值 92.45%）见
`outputs/acceptance_exp012_H20_20260824/`，**复现需要补回 `snn_torcs/condition.py`**
（仓库中是占位 stub，见 `docs/condition_module_spec.md`）；主 gate ① ② 不依赖它。

## 从零跑起来（Linux / Docker）

1. **装 TORCS**（本仓库不含仿真器本体）：按 `DEPLOY_TORCS.md` 编译安装
   vtorcs-RL-color（UDP 3101 + scr_server + g-track-1），并装 `xvfb x11-utils xautomation procps`。
2. **建 Python 环境**：`conda env create -f environment.yml && conda activate torcs_RL`
3. **交接包闭环验收**（= 跑通的判定标准）：
   ```bash
   bash run.sh        # 等价 bash snn_torcs_handoff/run.sh
   ```
   看到 per-step 数据逐行打印 + 末行 `TORCS terminated.` +
   `snn_torcs_handoff/data/replay_example.json` 重新生成 = 通过。
4. **出验收指标**：
   ```bash
   bash scripts/run_metric1_latency.sh     # 指标①（无需 TORCS，纯前向时延）
   bash scripts/run_metric2_standard_h20.sh  # 指标②（需要 TORCS）
   ```
   结果写入 `outputs/acceptance_exp012_H20_20260824/rerun/`。
   GPU 环境默认 cuda；无 GPU 机器：`DEVICE=cpu bash scripts/run_metric2_standard_h20.sh`。

## 目录结构

```
├── run.sh                        顶层 demo 入口（委托给交接包）
├── gym_torcs.py                  TORCS env（启动/Xvfb/生命周期管理）
├── snakeoil3_gym.py              UDP 客户端（端口 3101，SCR 协议）
├── autostart.sh                  xte 按键序列：自动进入 Practice 比赛
├── snn_torcs/                    模型与运行时
│   ├── models.py snn_core.py     FCANN / FCSNN(68→256→256→7)
│   ├── io.py                     checkpoint 读写（policy_net 解包）
│   ├── trainer.py                兼容 shim（仅 re-export io.py，无训练代码）
│   ├── env_wrapper.py            离散动作包装 + reward shaping
│   ├── run_mode.py               Xvfb :99 + practice/screen.xml 备份改写恢复
│   ├── condition.py              ⚠️ 占位 stub（矩阵复现需补，见 docs/）
│   └── utils.py
├── configs/default.yaml          网络/环境/评测参数
├── conditions/conditions.yaml    6+1 条件预设（供 condition.py 使用）
├── scripts/                      验收复现入口（metric1 / metric2 / 7条件矩阵）
├── environment.yml               conda 环境（python 3.13 + torch 2.12 + gym 0.26.2）
├── snn_torcs_handoff/            原样冻结的交接子包（自洽，含示例模型 exp013）
├── outputs/acceptance_exp012_H20_20260824/   原样冻结的验收证据包（勿改）
└── docs/                         部署指南 / condition 重建规格 / practice.xml 模板 / 移植核查报告
```

## 权重文件（sha256）

| 文件 | sha256 | 用途 |
|---|---|---|
| `outputs/.../models/exp012_ours_t4_best.pth` | `61c9982f…c42f2d7b5e25` | ours SNN T=4（指标①②主角） |
| `outputs/.../models/exp013_baseline_t8_best.pth` | `22825d7b…2936fe6e9dac0` | baseline T=8（时延对照） |
| `snn_torcs_handoff/model/exp013_best.pth` | 同上（与 baseline 同一文件） | demo 示例模型 |

checkpoint 为 trainer 全量格式（含 `policy_net` 键），由 `snn_torcs/io.py::load_state_dict`
解包加载；已实测与 `FCSNN(68,256,7)` 结构逐层匹配。

## 已知事项

- **alt-track 条件**：原环境跑完整 1000 步会在 ~556 步触发 TORCS CommTimeout（证据在
  `outputs/.../runtime_notes/`）；只能以 pre-timeout 段（H20=93.71%）作为辅助证据。
- **gym 0.26.2 + numpy 2.x** 会打印 deprecation 警告，实测不影响运行（原环境同组合）。
- 归档目录（`outputs/acceptance_.../`、`snn_torcs_handoff/`）保持原样不动；新证据一律写
  `outputs/.../rerun/`（已 gitignore）。
- Windows 协作者注意：`.gitattributes` 已强制 LF，请勿改动，否则 .sh 会在 Linux 上炸。

## 更多文档

- `DEPLOY_TORCS.md` — TORCS 在 docker 上的完整安装部署指南
- `AGENTS.md` — 给部署用 AI agent（pi）的操作说明与验收 gate
- `docs/CROSS_VALIDATION_REPORT.md` — 移植前对两个原始包的依赖交叉核查报告
- `docs/condition_module_spec.md` — condition.py 重建规格
- `docs/practice.g-track-1.xml` — scr_server + g-track-1 的 practice.xml 模板
