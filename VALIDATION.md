# CP3 Notebook 验证记录

验证日期：2026-08-31（Asia/Shanghai）

## 验证矩阵

| 文件 | JSON / AST | 模拟 LLM + 内存检查点 | 模拟 LLM + 对应检查点后端 | 结果断言 |
| --- | --- | --- | --- | --- |
| `CP3/04_error.ipynb` | 通过 | 通过 | 通过（PostgreSQL） | 捕获预期 `expected_error`；失败任务写入检查点 |
| `CP3/05_find_error.ipynb` | 通过 | 通过 | 通过（PostgreSQL） | 读取 `step=1`，看到 `node_poem` 有结果、`node_joke` 有错误 |
| `CP3/06_fix_error.ipynb` | 通过 | 通过 | 通过（PostgreSQL） | `resume_from=('node_joke',)`，最终生成 `final_output` |
| `CP3/07_replay.ipynb` | 通过 | 通过 | 通过（PostgreSQL） | 找到并行 checkpoint，回放后生成 `final_output` |
| `CP3/08_fork.ipynb` | 通过 | 通过 | 通过（InMemorySaver） | 两种 `update_state` 都生成新检查点并走到预期分支 |

## 具体证据

- Python 环境：Conda `LangChain`，Python 3.11.15。
- 依赖版本：`langgraph 1.2.9`、`langgraph-checkpoint-postgres 3.1.2`、`langchain-core 1.5.0`、`langchain-deepseek 1.0.1`、`python-dotenv 1.2.2`、`loguru 0.7.3`、`nbclient 0.10.4`、`nbformat 5.10.4`。
- PostgreSQL：本机 `localhost:5432/langgraph_db` 可连接；`checkpointer.setup()` 后存在 `checkpoints`、`checkpoint_blobs`、`checkpoint_writes`、`checkpoint_migrations` 四张表。
- Notebook 结构：五个目标文件均为 nbformat 4；代码单元通过 Python AST 解析；这五个 Notebook 都加入了中文 Markdown 说明，且没有提交执行输出或 base64 图片。另对既有 `CP3/02_in_SQL.ipynb` 做了同样的 `LANGGRAPH_DB_URL` 环境变量迁移，仓库不再硬编码示例数据库密码。
- SVG：远程仓库 `assets/img/` 的 8 张 Day4 图与 2 张 Day6 图均通过 XML 解析。
- 链接：README、Day4、Day6 的仓库相对链接均解析到现有文件；关键 GitHub、博客与 LangGraph 官方文档链接返回 HTTP 200。
- 缓存：远程仓库没有 `.ipynb_checkpoints/`、`.jupyter_cache/`、`__pycache__/` 或临时审计文件。

## 外部依赖限制

本次自动验证使用**假 LLM**替代真实 DeepSeek 请求，原因是当前进程没有 `DEEPSEEK_API_KEY`，且不应在仓库中写入真实凭据。PostgreSQL 使用本机真实服务，因此验证覆盖了检查点建表、写入、读取、恢复、Replay 与 Fork 的 LangGraph 运行路径。

在用户自己的环境中执行时还需要：

1. 按 `.env.example` 配置真实 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 和 `LANGGRAPH_DB_URL`。
2. 确保 DeepSeek 网关支持 `with_structured_output`，否则 CP3-08 的路由模型需要换成支持结构化输出的模型或适配器。
3. 按 CP3-04 → CP3-05 → CP3-06 的顺序使用同一个 `CP3_ERROR_THREAD_ID`；CP3-07 使用自己的线程，CP3-08 使用内存检查点。
4. 运行 CP3-04 时看到 `expected_error` 属于教学预期；若看到缺少环境变量、连接拒绝或模型鉴权异常，则属于本地配置问题。

## 未执行项

- 未在本次环境中调用真实 DeepSeek API，避免产生费用、依赖用户密钥或把网关可用性误当作代码验证。
- 未修改现有 CP1、CP2、CP3-01～03 的历史输出；本次仅维护 CP3-02 的凭据配置、用户指定的 CP3-04～08，并清理这五个新 Notebook 的历史输出。
