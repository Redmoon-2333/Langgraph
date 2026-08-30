# Langgraph — GuiGU 课程实战（CP1-CP3）

> 图驱动 Agent：State / Reducer / Control Flow / Checkpointer / Replay / Fork 全景实现

课程来源：尚硅谷 LangGraph（CP3 错误处理、检查点与时间旅行，对应第 76 集前后内容）
在线博客：https://redmoon-2333.github.io

## 目录

- `Day4.md`：CP1–CP3 状态、控制流、持久化与时间旅行总览
- `Day6.md`：CP3-04–08 错误恢复、Replay 与 Fork 学习笔记
- `assets/img/`：Day4 与 Day6 的流程图、时序图和状态图
- `CP1/`：8 个 Notebook，状态底座
- `CP2/`：15 个 Notebook，控制流与运行时治理
- `CP3/`：8 个 Notebook，检查点、错误恢复、Replay 与 Fork

当前仓库共 **31 个 Notebook**。

## CP3 学习路线

| 顺序 | Notebook | 学习主题 | 关键 API / 现象 |
|---:|---|---|---|
| 1 | `CP3/01_in_memory.ipynb` | 内存检查点 | `InMemorySaver`、`thread_id` |
| 2 | `CP3/02_in_SQL.ipynb` | PostgreSQL 检查点 | `PostgresSaver`、`setup()` |
| 3 | `CP3/03_history_state.ipynb` | 快照与历史 | `StateSnapshot`、`get_state_history()` |
| 4 | `CP3/04_error.ipynb` | 制造并观察错误 | 并行超步、失败任务、可恢复检查点 |
| 5 | `CP3/05_find_error.ipynb` | 定位错误 | `values`、`next`、`tasks[].result/error` |
| 6 | `CP3/06_fix_error.ipynb` | 修复后续跑 | `invoke(None, config)`、成功结果复用 |
| 7 | `CP3/07_replay.ipynb` | 历史回放 | 指定 `checkpoint_id` 重放剩余节点 |
| 8 | `CP3/08_fork.ipynb` | 修改历史并分支 | `update_state()`、`as_node=START/节点名` |

### CP3-04 → CP3-06：错误三部曲

这三个 Notebook 共享 `CP3_ERROR_THREAD_ID`（默认 `chapter03-05`），必须按顺序执行：

1. `04_error.ipynb` 在 `node_joke` 中故意抛出异常。这个异常是**预期教学现象**，代码会捕获并打印摘要，但失败超步仍会写入 PostgreSQL。
2. `05_find_error.ipynb` 不重新执行图，只读取同一线程的 `StateSnapshot`，从 `tasks` 中观察哪个节点成功、哪个节点带有 `error`。
3. `06_fix_error.ipynb` 移除故障注入，调用 `graph.invoke(None, config)`，从失败检查点续跑未完成任务。

### CP3-07：Replay

`07_replay.ipynb` 使用 `CP3_REPLAY_THREAD_ID`（默认 `chapter03-08`）运行两次，然后按 `next` 找到 `node_poem` 与 `node_joke` 即将执行的历史快照。指定该快照的 `config` 调用 `invoke(None, ...)`，会重新调用两个 LLM 节点；因此 Replay 保证状态起点和执行拓扑，不保证模型文本完全相同。

### CP3-08：Fork

`08_fork.ipynb` 使用 `InMemorySaver`，不依赖 PostgreSQL，但仅在当前 Python 进程有效。它从路由节点之前的快照分出两条未来：

- `as_node=START`：修改 `user_input`，续跑时重新调用结构化路由模型。
- `as_node="router_node"`：直接伪造 `topic/mode` 作为路由节点产出，跳过一次 LLM 调用，但仍会重新计算条件出边。

## 环境与安装

- Python **3.11+**
- PostgreSQL（CP3-02、04–07 使用）
- DeepSeek API（调用 LLM 的 Notebook 使用）
- LangGraph 1.x 已验证；代码使用的多 schema 关键字参数至少需要较新的 LangGraph 版本

推荐在仓库根目录创建虚拟环境并安装：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

也可以直接安装核心依赖：

```bash
pip install langgraph langchain-core langchain-deepseek \
  langgraph-checkpoint-postgres "psycopg[binary]" \
  python-dotenv loguru jupyterlab
```

## 配置

复制 `.env.example` 为 `.env`，再填写真实配置。`.env` 已被 Git 忽略，不要把 API Key 或数据库密码提交到仓库。

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_MODEL=deepseek-v4-flash
LANGGRAPH_DB_URL=postgresql://langgraph_user:修改后的密码@localhost:5432/langgraph_db?sslmode=disable
```

启动 Jupyter 时建议位于仓库根目录，使 Notebook 中的 `load_dotenv(override=True)` 能找到 `.env`：

```bash
jupyter lab
```

### PostgreSQL 初始化

先创建数据库和用户，再运行任意使用 `PostgresSaver` 的 Notebook。每个 Notebook 内的 `checkpointer.setup()` 会幂等初始化 LangGraph 检查点表：

```sql
CREATE USER langgraph_user WITH PASSWORD '修改后的密码';
CREATE DATABASE langgraph_db OWNER langgraph_user;
```

不要把上面的示例密码直接用于生产环境；本项目连接串只用于本地教学。

## 执行顺序与验证提示

- CP1 → CP2 → CP3 是知识顺序；CP3 内部推荐 `01 → 02 → 03 → 04 → 05 → 06`，然后再做 `07 → 08`。
- `04_error` 的异常是故意制造的；看到 `expected_error` 不代表 Notebook 失败。
- `05_find_error` 与 `06_fix_error` 依赖 CP3-04 写入的同一 `thread_id`。如果该线程已经恢复到 `END`，请设置新的 `CP3_ERROR_THREAD_ID` 并重新执行 04。
- `PostgresSaver` 必须在 `with` 块内使用；离开上下文后连接会关闭。
- `08_fork` 只使用内存检查点，重启 Kernel 后需要从第一个代码单元重新执行。
- 输出已清理为可复现的教学输出，避免把大段 LLM 文本和图像 base64 当成 Notebook 内容提交。

## 对应笔记

- [Day4：图驱动 Agent 的状态、控制流与记忆回放](Day4.md)
- [Day6：错误处理、Replay 与 Fork](Day6.md)
- [线上博客](https://redmoon-2333.github.io)
- [LangGraph 官方 Time Travel 文档](https://docs.langchain.com/oss/python/langgraph/use-time-travel)

## 免责声明

示例使用 LLM 生成诗歌和笑话，仅用于展示图执行、检查点和分支机制。真实业务中应增加超时、重试、幂等、日志脱敏、凭据管理和数据库备份策略。

署名：RedMoon
