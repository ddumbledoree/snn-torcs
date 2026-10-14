================================================================================
SNN-TORCS 展示软件 · 交接包
================================================================================
本包提供“推理运行包”,供展示软件开发。TORCS 仿真环境由贵方自备(已装)。

目录:
  README.txt                 本文件(跑通指南)
  docs/manifest.txt          展示软件设计稿(软件要做成什么样、接口字段)
  run.sh                     接口验收脚本(一键跑通)
  environment.yml            conda 依赖
  app/                       运行目录:推理代码 + autostart.sh + configs
    ├── eval_steptrace.py    逐帧 eval(实时 demo + 生成回放 JSON)
    ├── autostart.sh         自动进 practice 比赛(依赖 xte)
    ├── gym_torcs.py / snakeoil3_gym.py / snn_torcs/  推理与环境对接
    └── configs/default.yaml 网络与运行参数
  model/exp013_best.pth      示例模型(baseline SNN, T=8)
  data/replay_example.json   示例逐帧回放数据(step-level,回放模式输入)
  reports/                   验收指标证据
    ├── latency_benchmark.json   指标① 决策速度(41.6%)
    └── eval_exp013.json         示例模型 eval / 模型卡

--------------------------------------------------------------------------------
前置依赖(贵方环境需具备)
--------------------------------------------------------------------------------
  - TORCS:已装,且 ~/.torcs/config/raceman/practice.xml 指向 g-track-1(road)
  - conda
  - X11 自动化工具(闭环 headless 启动需要):
      xte      -> sudo apt install xautomation
      Xvfb     -> sudo apt install xvfb
      xdpyinfo -> sudo apt install x11-utils

--------------------------------------------------------------------------------
跑通步骤(接口验收)
--------------------------------------------------------------------------------
  1. conda env create -f environment.yml && conda activate torcs_RL
     (若已建好环境,只需 conda activate torcs_RL)
  2. ./run.sh
  3. 看到 per-step 数据逐行打印 + 末行 “TORCS terminated.” +
     data/replay_example.json 生成 = 验收通过

  run.sh 等价于:
    cd app && python eval_steptrace.py --model-type snn_baseline \
      --checkpoint ../model/exp013_best.pth --max-steps 200 --run-mode fast_train \
      --report ../data/replay_example.json

--------------------------------------------------------------------------------
验收通过后
--------------------------------------------------------------------------------
  展示软件设计需求见 docs/manifest.txt(三栏布局、两种模式、字段清单、验收指标)。
  关键接口:
    - 实时模式:在 app/ 内加载 .pth → model(state) 得 7 维 Q → argmax 得动作;
      env.step 返回的 info 含 speedX/trackPos/angle/distRaced/reward。
    - 回放模式:读 data/replay_example.json 的 runs[].steps[](每帧一个 record)。

  说明:示例模型 exp013 是 baseline(T=8),接口稳定;最终 ours(T=4)模型到位后
        替换 model/*.pth 即可,代码与数据格式不变。
================================================================================
