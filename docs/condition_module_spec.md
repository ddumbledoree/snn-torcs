# `snn_torcs/condition.py` — 接口规格（供重建使用）

> **重建完成（2026-10-14）**：按本规格实现了备份→原地改写→恢复→污染门方案，
> 全矩阵重跑与归档值逐位一致（见 rerun/condition_matrix_h20/）。
> 实现时发现的两个部署差异：(1) 本容器的 TORCS 从**系统级**
> /usr/local/share/games/torcs/config/raceman/practice.xml 读赛道，
> practice.xml 必须 user+system 两份同步补丁（spec 原假设只改 ~/.torcs）；
> (2) e-track-1 主路面 surface 名 asphalt-bw3 已验证存在（stock friction 1.2，
> 预设改为 1.15 与归档行为一致）。

> 背景：该模块在原私有训练仓库（`/home/desc/gym_torcs/snn_torcs/condition.py`）中实现，
> 未包含在两个交接包里。本文档从 `conditions/conditions.yaml` 头注释与
> `scripts/eval_condition.py` 的调用点反推其接口契约，供：
> (a) 从原机器取回原文件后对照确认，或 (b) 在远端重建实现。

## 1. 对外接口（eval_condition.py 的全部调用点）

```python
from snn_torcs import condition as cond_mod

condition = cond_mod.load_condition(name)        # name: 预设名，如 "wet-all"
before = cond_mod.snapshot_stock()               # 跑前快照
condition.apply()                                # 切换 TORCS 配置到该条件
cond_mod.install_signal_handlers(condition, before)   # SIGINT/SIGTERM 时恢复
...
condition.restore()                              # 跑完恢复
cond_mod.assert_stock_unchanged(before)          # 污染门：stock 文件必须逐字节还原
```

对象属性：
- `condition.spec` — dict，来自 conditions.yaml 的预设段，至少含
  `track` / `category` / `car` / `friction`（friction 为 `[{surface, value}, ...]`）

## 2. 条件预设与它需要改的三处 TORCS 配置

| 预设 | track | car | friction 改动 |
|---|---|---|---|
| baseline（回归锚点） | g-track-1 (road) | car1-trb1 | asphalt-g1 = 1.15（=原厂值，等价不改） |
| wet | g-track-1 | car1-trb1 | asphalt-g1 = 0.7 |
| icy | g-track-1 | car1-trb1 | asphalt-g1 = 0.4 |
| wet-all | g-track-1 | car1-trb1 | asphalt-g1 / -left / -right / -patch = 0.6 |
| alt-car | g-track-1 | **car2-trb1** | asphalt-g1 = 1.15 |
| alt-track | **e-track-1 (road)** | car1-trb1 | asphalt-bw3 = 1.15（e-track-1 主路面，yaml 注明需验证） |

三处配置文件与编辑方式（yaml 头注释原文）：
1. **赛道选择** — `~/.torcs/config/raceman/practice.xml` 的
   `<section name="Tracks"><section name="1"><attstr name="name" val="..."/><attstr name="category" val="road"/>`
   （scoped edit，只动这一段）
2. **车辆选择** — `scr_server.xml` 的 `<attstr name="car name" val="...">`（global，安全）
3. **摩擦** — 赛道 XML（如 `/usr/local/share/games/torcs/tracks/road/g-track-1.xml`）中
   `<section name="asphalt-g1"><attnum name="friction" .../>`（scoped，只改该 surface 段）

## 3. "cwd symlink-farm" 零写入语义（原实现方式，重建时的行为目标）

- 目标：**运行期间不写任何 stock 文件**（连临时改写都不允许），结束后 stock 逐字节不变。
- 原实现的思路（据 yaml 注释"cwd symlink-farm"）：在工作目录建一个 symlink 农场——
  绝大多数文件是指向 stock 的软链，仅将要覆盖的 2-3 个文件（practice.xml / scr_server.xml /
  赛道 XML）替换为改写后的实拷贝——并让 TORCS 从农场读取配置启动。
- **重建时允许退化为等价方案**：apply() 前备份 → 原地改写 → restore() 时从备份还原 →
  污染门校验。注意：`run_mode.py` 已有同类逻辑（`backup_configs/restore_configs` 对
  practice.xml/screen.xml 的 .bak 机制），语义一致，可参考或复用其 `_patch_xml_attr`。
- 污染门实现建议：对上述 3 类文件（及其可能触碰的所有 stock 路径）跑前/跑后各算一次
  sha256，`assert_stock_unchanged` 比对，不一致 raise。

## 4. 验证方式（重建后）

1. `--condition baseline` 跑 200 步：H20 与 `scripts/eval_steptrace.py` 直跑（stock）结果应一致
   （yaml 注释：baseline 预设"reproduces stock exactly"，是回归锚点）。
2. `--condition wet` 后检查赛道 XML 的 friction 段确实为 0.7（改写生效），
   结束后 `diff` stock 无差异（污染门 PASS 会打印）。
3. `run_7_condition_h20_matrix.sh` 全矩阵重跑，与
   `outputs/acceptance_exp012_H20_20260824/reports/condition_matrix/` 归档值对照
   （允许随机性波动，H20 应在同一水平：6 条件均值 92.45%）。

## 5. 已知坑

- e-track-1 的主路面 surface 名 `asphalt-bw3` 是 yaml 注释里写的"verify in test"项，
  重建后要先打开 e-track-1.xml 确认实际 section 名。
- alt-track 条件在原环境就存在已知问题：跑到 ~556 步触发 TORCS CommTimeout
  （见 `outputs/.../runtime_notes/alt-track_t0.15_verbose1000_commtimeout.log`），
  验收时只报 pre-timeout 段（H20=93.71%），不得当作完整 1000 步通过。
- scr_server.xml 的具体路径随 TORCS 变体可能不同（`~/.torcs/config/scr_server.xml` 或
  robots 目录下），实现时应搜索候选路径并在找不到时明确报错。
