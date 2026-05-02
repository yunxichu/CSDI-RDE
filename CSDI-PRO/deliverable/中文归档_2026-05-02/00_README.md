# 中文归档 — 2026-05-02

**归档版本**：`013d766`（branch `csdi-pro-m3-alt`，已同步 GitHub `origin`）
**论文标题**：受损混沌上下文中的可预测性前沿（Forecastability Frontiers Under Corrupted Chaotic Contexts）
**投稿目标**：NeurIPS / ICLR 2026

---

## 本归档包含

| 文件 | 内容 | 用途 |
|:--|:--|:--|
| `00_README.md` | 本索引 | 归档导览 |
| `01_论文_中文.md` | 当前 locked 中文主稿（清洁副本，与 `paper_draft_zh.md` 一致）| 阅读 / 评审 / 翻译 LaTeX |
| `02_实验记录_中文.md` | 17 项实验的逐项记录（动机、配置、关键数字、源 JSON）| 复核数字、回查实验 |
| `03_技术流程_中文.md` | 完整技术流程（数据生成 → corruption → imputation → forecasting → metric → 训练 → 复现）| 工程交接、代码导览 |

## 与上层文件的关系

| 本目录文件 | 对应的 master 文件 |
|:--|:--|
| `01_论文_中文.md` | `deliverable/paper/paper_draft_zh.md` |
| `02_实验记录_中文.md` | `deliverable/EXPERIMENTS_FULL_ZH.md` |
| `03_技术流程_中文.md` | （本归档新写，无 master 对应文件）|

如需最权威版本（commit-tracked），请直接看上一栏的 master 文件。本目录的副本是一份"freeze 时刻"的可移植版本。

## Quick facts

- **论文核心主张**：稀疏观测对预训练混沌 forecaster 制造尖锐可预测性前沿；corpus-pretrained 结构化 imputation 是穿过 transition band 的杠杆。
- **主要系统**：Lorenz-63（headline）、Lorenz-96 N=20（cross-system）、Rössler、Kuramoto；scope boundary：Mackey-Glass / Chua（合成 structure 违反）+ Jena Climate（真实 periodic-dominant）
- **主要 forecaster**：Panda-72M（headline，混沌专用预训）、Chronos-bolt-small（broad time-series，跨 foundation 控制）、DeepEDM（延迟坐标 companion）、EnKF（已知动力学上界）
- **主要 imputer**：linear、AR-Kalman、CSDI（headline）、SAITS-pretrained（替代 imputer 控制）
- **17 个实验**全部用 v2 corruption 协议（少数 legacy S0–S6 协议在 Appendix B 标注为辅助方向证据）

## 严苛审稿人 6 issue 闭合状态

| # | Issue | 状态 | Commit |
|:-:|:--|:--|:--|
| 1 | Jena 单 forecaster 攻击 | ✅ 闭合 | `22c0904` 加 cross-forecaster + clean upper |
| 2 | §4.2 mechanism CSDI-specific？| ✅ 闭合 | `f95e948` 加 SAITS arm |
| 3 | §4.1 legacy S0–S6 在主文 | ✅ 闭合 | `22c0904` 移到 Appendix |
| 4 | §4.4 drift callout 在主文 | ✅ 闭合 | `22c0904` 移到 Appendix B.3 |
| 5 | decoder probe open hypothesis | ✅ 闭合 | `6189a12` per-layer probe → mid-encoder convergence |
| 6 | MG/Chua scope coverage | ✅ 部分闭合 | `013d766` §6.3 ↔ §6.6 scope boundary 桥接 |

## Spotlight 概率自评：~50–55%；Poster ~85–90%

---

## 文档导览

**第一次读论文**：从 `01_论文_中文.md` 开始（阅读时间约 30 分钟）

**复核某个数字**：到 `02_实验记录_中文.md` 找对应实验编号 → 看源 JSON 路径

**接手代码 / 复现实验**：先看 `03_技术流程_中文.md` § 1（系统概览） → § 9（复现指引）

**LaTeX 转换前**：看 `01_论文_中文.md` 的英文兄弟 `deliverable/paper/paper_draft_en.md`（投稿就用英文版）
