# DEPLOY_TORCS.md — 在 Linux/Docker 上从零部署 TORCS + 本仓库

目标读者：在远程无头 Linux 服务器（docker）上做部署的人或 AI agent。
唯一目标：`bash run.sh` 跑通 → `run_metric1/2` 复现指标。

## 0. 基底镜像建议

```dockerfile
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
```

vtorcs-RL-color 是 ~2013 年的 C++ 代码，**不要用太新的发行版**（gcc 过新会出一堆编译错）。
Ubuntu 22.04 实测口径最稳；若编译报错多，降到 20.04。

## 1. 系统依赖（一次装齐）

```bash
apt-get update && apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool pkg-config \
    libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev \
    libxrender-dev libxrandr-dev libpng-dev \
    libgl1-mesa-dri \
    xvfb x11-utils xautomation procps \
    git wget ca-certificates
```

对应关系（代码里实际用到的）：`Xvfb`(虚拟显示 :99)、`xdpyinfo`(x11-utils，探活)、
`xte`(xautomation，autostart.sh 自动点菜单)、`pkill`(procps，杀残留 torcs)。

## 2. 编译安装 vtorcs-RL-color（必须这个变体，端口 3101）

本仓库 `snakeoil3_gym.py` 连 **UDP 3101**（标准 TORCS 的 SCR 端口是 3001），
且需要 `scr_server` 机器人驱动 —— 即 `ugo-nama-kun/gym_torcs` 配套的
**vtorcs-RL-color**（README 明确点名此变体）。

```bash
git clone https://github.com/ugo-nama-kun/gym_torcs.git
cd gym_torcs/vtorcs-RL-color
./configure          # 显卡/编译问题时可试 ./configure --disable-xrandr
make -j"$(nproc)"
make install         # 需 root；装到 /usr/local 下
make datainstall     # 装赛道/车辆数据（g-track-1、e-track-1、car1-trb1、car2-trb1 都在其中）
```

编译注意（原 README 要点）：
- 依赖 plib **1.8.5**（ubuntu 的 `libplib-dev` 即此版本系）；若自编译 plib，64 位必须 `-fPIC`。
- 新 gcc 下若报隐式函数声明/窄化错误，优先降 ubuntu 版本而不是大改代码。

验证安装：

```bash
command -v torcs            # /usr/local/bin/torcs
ls /usr/local/share/games/torcs/tracks/road/ | grep -E 'g-track-1|e-track-1'
ls /usr/local/share/games/torcs/cars/models/ | grep -E 'car[12]-trb1'
```

## 3. 生成用户配置 + 写入 practice.xml（scr_server + g-track-1）

TORCS 首次启动才生成 `~/.torcs`：

```bash
timeout 15 xvfb-run -a torcs || true
ls ~/.torcs/config/         # 应有 raceman/、screen.xml 等
```

然后安装本仓库的模板（已把赛道改为 g-track-1、驱动设为 scr_server，
基于上游参考 practice.xml）：

```bash
mkdir -p ~/.torcs/config/raceman
cp docs/practice.g-track-1.xml ~/.torcs/config/raceman/practice.xml
```

> 替代法：`xvfb-run -a torcs` 起来后用 GUI（Race → Practice → Configure）选
> g-track-1 + scr_server，效果等价；模板法更快。

运行期 `snn_torcs/run_mode.py` 会自动：备份 practice.xml/screen.xml →
切 `display mode="results only"`、分辨率 64x64（fast_train 无渲染加速）→ 结束后恢复。
不需要手工干预。

## 4. Python 环境

```bash
# 装 miniconda/micromamba 后：
conda env create -f environment.yml
conda activate torcs_RL
python -c "import torch, yaml, numpy, gym; print('ok')"   # gym 会打 deprecation 警告，正常
```

- 有 GPU 的 docker：需 NVIDIA Container Toolkit，`--device cuda` 直接用。
  ⚠️ **驱动版本门槛**：environment.yml 钉的是 torch 2.12 + CUDA 13 栈，要求宿主机 NVIDIA
  驱动非常新（≥580 系）。`nvidia-smi` 看驱动，若不满足：直接全程 `DEVICE=cpu`（指标①②都能跑，
  只是绝对时延数值不同），或自建 cu12 版 torch 环境。**不要在这个卡点上消耗部署主线时间。**
- 纯 CPU docker：脚本用 `DEVICE=cpu` 覆盖（见 §5）。environment.yml 会照装 CUDA 版 torch
  wheel（体积大但 CPU 可跑）；追求轻量可自建 cpu-only env（torch cpu wheel + pyyaml + numpy<2.5 + gym==0.26.2）。
- 网络慢/受限时：conda 与 pip 均可配国内镜像（如清华源）再 `conda env create -f environment.yml`，
  cuda-toolkit 相关 wheel 合计数 GB，镜像能省大量时间。

## 5. 验收 gate（逐级解锁，每步过了再下一步）

```bash
# G1 仿真器链路（不用模型）
command -v torcs Xvfb xdpyinfo xte pkill

# G2 纯推理冒烟（不用 TORCS；--help 能打印说明 import 链完整）
python scripts/benchmark_policy_latency.py --help
DEVICE=cpu python scripts/benchmark_policy_latency.py \
    --baseline-ckpt outputs/acceptance_exp012_H20_20260824/models/exp013_baseline_t8_best.pth \
    --ours-ckpt     outputs/acceptance_exp012_H20_20260824/models/exp012_ours_t4_best.pth \
    --report-dir /tmp/g2 --device cpu
#   → 期望 latency_reduction > 0.10（归档值 0.4258；CPU 上绝对值不同但方向应一致）

# G3 闭环 demo（TORCS + 模型，= 移植成功判定）
bash run.sh
#   → per-step 逐行打印 + "TORCS terminated." + data/replay_example.json 更新

# G4 验收指标复现
bash scripts/run_metric1_latency.sh                  # 指标①（无 GPU: DEVICE=cpu）
bash scripts/run_metric2_standard_h20.sh             # 指标② H20 ≥ 0.90（无 GPU: DEVICE=cpu）
#   结果在 outputs/acceptance_exp012_H20_20260824/rerun/ 下

# G5（可选）7 条件鲁棒性矩阵 —— 先补 snn_torcs/condition.py（见 docs/condition_module_spec.md）
bash scripts/run_7_condition_h20_matrix.sh
```

## 6. 运行纪律与预期管理

- **一次只跑一个评测进程**：Xvfb :99 和 UDP 3101 是全局唯一资源，两个 eval 并行会互相杀对方的
  torcs / 抢端口。矩阵脚本内部已经是串行的，不要同时手动再起 run.sh。
- **复现数字的语义**：TORCS 在新机器上重新编译后，物理仿真与原环境存在微小差异，
  H20 不会精确复刻 90.50%，时延绝对值也会不同。按 **gate 口径**判读（指标① reduction≥0.10、
  指标② H20≥0.90）而不是逐位对齐归档值；若 H20 略低于 0.90（如 88~90%），属于证据解释问题，
  优先多跑几遍看均值/检查 TORCS 构建差异，而不是怀疑代码移植错了。

## 7. 故障排查速查

| 症状 | 原因 | 处置 |
|---|---|---|
| `Waiting for server on 3101...` 死循环 | torcs 没起 / 端口不是 3101（装成了标准 TORCS）/ practice.xml 没选 scr_server | 确认装的是 vtorcs-RL-color；重装 §3 的 practice.xml；`ss -ulnp | grep 3101` |
| `Xvfb failed to start on :99` | xvfb 未装 / :99 被占 | `apt install xvfb`；`pkill Xvfb` 后重试 |
| `xte: command not found` | xautomation 未装 | `apt install xautomation` |
| autostart 按键没生效（黑屏不动） | xte 时序、locale 键盘映射 | 检查 DISPLAY=:99；手动 `xte 'key Return'` 验证 |
| CommTimeout（中途断） | TORCS 内存泄漏老 bug / 环境不稳 | 已有 relaunch 机制（relaunch_interval=3）；alt-track 条件下 ~556 步超时是已知问题 |
| git clone 后 .sh 报 `\r` | CRLF 污染 | `.gitattributes` 已强制 LF；勿改 |
| `python: No module named gym` | 没激活 torcs_RL | `conda activate torcs_RL`，或 `PY=<env>/bin/python` 覆盖 |
