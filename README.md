# Qwen3-Embedding：自托管 Embedding 推理服务

把 `Qwen/Qwen3-Embedding-4B` 封装成一个可自托管的 embedding 推理服务：对外提供 OpenAI 兼容的 `POST /v1/embeddings`、HTTP MCP Server、内置调试页面，并附带 FastAPI 的交互式接口文档，方便在内网或私有环境里快速接入与运维。

## 功能
- OpenAI 兼容 Embeddings API：`POST /v1/embeddings`
- Qwen 检索增强字段：`input_type=query|document`、`instruction`
- MCP Server：HTTP 挂载到 `POST/GET /mcp`（Streamable HTTP）
- 内置 Web UI：`GET /`（可直接输入文本调试 embedding）
- 交互式接口文档：`GET /docs`（Swagger UI）与 `GET /redoc`
- 模型自动下载与缓存：将 `./models` 挂载到容器 `/models`（Hugging Face 缓存目录）
- 运维友好：健康检查 `GET /health`；可选热重载 `POST /admin/reload`（`ADMIN_TOKEN` 保护）
- 容器内双层架构：外层 FastAPI，对内拉起 `vLLM` 子进程做实际推理

## 快速开始
```bash
docker compose up -d --build
```

如果机器需要走代理才能访问 Hugging Face，可在同目录创建 `.env`（或启动前导出环境变量）：
```bash
HTTP_PROXY=http://127.0.0.1:7890
# 可选：不走代理的地址（默认：localhost,127.0.0.1）
# NO_PROXY=localhost,127.0.0.1
```

打开：
- Web UI：http://localhost:12302/
- MCP HTTP：http://localhost:12302/mcp
- 接口文档（Swagger）：http://localhost:12302/docs
- 接口文档（ReDoc）：http://localhost:12302/redoc
- 健康检查：http://localhost:12302/health

## 架构说明
- **对外端口**：`PORT=12302`
- **容器内 vLLM**：默认监听 `127.0.0.1:8001`
- **工作方式**：外层 FastAPI 接收请求，必要时注入 Qwen query instruction，然后转发给容器内的 `vLLM` 子进程
- **自动下载**：首次启动时若本地缓存不存在模型，`vLLM` 会自动从 Hugging Face 拉取 `MODEL_ID`

这意味着你平时只需要访问：

```text
http://localhost:12302/v1/embeddings
```

不需要直接访问容器内的 `8001` 端口。

## OpenAI SDK 快速开始

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12302/v1",
    api_key="dummy",
)

response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-4B",
    input="What is the capital of China?",
    extra_body={
        "input_type": "query",
        "instruction": "Given a web search query, retrieve relevant passages that answer the query",
        "dimensions": 1024,
    },
)

print(len(response.data[0].embedding))
```

## curl 示例
```bash
curl http://localhost:12302/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": [
      "What is the capital of China?",
      "The capital of China is Beijing."
    ],
    "input_type": "query",
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "dimensions": 1024
  }'
```

## 接口一览
- `POST /v1/embeddings`
  - 标准字段：`input`、`model`、`dimensions`、`encoding_format`、`user`
  - 扩展字段：`input_type`、`instruction`
- `POST /mcp` / `GET /mcp`：MCP Streamable HTTP 入口
- `GET /docs` / `GET /redoc`：交互式接口文档
- `GET /openapi.json`：OpenAPI 规范 JSON
- `GET /health`：健康检查与运行参数
- `POST /admin/reload`：热重载模型（需 `x-admin-token`）

## Qwen 扩展语义
- `input_type=query` 时，服务会把每条输入包装为：

```text
Instruct: {instruction}
Query:{text}
```

- 若未传 `instruction`，使用默认英文检索 instruction：`Given a web search query, retrieve relevant passages that answer the query`
- `input_type=document` 或未传 `input_type` 时，不自动改写输入
- 单次请求内的所有输入共享同一个 `input_type` 和 `instruction`
- `dimensions` 支持 `32-2560`

## MCP 快速开始

### HTTP MCP
服务启动后，MCP Streamable HTTP 入口固定为：

```text
http://localhost:12302/mcp
```

适合远端客户端或通过网关统一接入的场景。

## MCP 能力一览

### Tool
- `embed_text`
  - 入参：`texts`（必填，字符串或字符串数组）、`input_type`（可选）、`instruction`（可选）、`dimensions`（可选）
  - 返回：标准 OpenAI embeddings 响应形状

### Resources
- `qwen3embedding://health`：当前模型、端口、backend 就绪状态、默认 instruction 等
- `qwen3embedding://usage`：MCP 工具参数说明与使用建议

### Prompts
- `retrieval_embedding_workflow`：指导客户端如何区分 query/document，并在 query 侧传入 instruction

## 模型自动下载与缓存
- 容器内默认设置 `HF_HOME=/models`
- `docker-compose.yml` 默认挂载 `./models:/models`
- 如果本地没有缓存，首次启动会自动下载模型
- 如果已下载过，后续重启会复用缓存，不会重复下载

## Docker 部署示例
```bash
docker run -d --name qwen3_embedding_openai \
  --gpus all \
  -p 12302:12302 \
  -e MODEL_ID="Qwen/Qwen3-Embedding-4B" \
  -e NVIDIA_VISIBLE_DEVICES="0" \
  -e HF_HOME="/models" \
  -v ./models:/models \
  qwen3-embedding-openai:latest
```

## 切换模型（需重启）
在 `docker-compose.yml` 中修改 `MODEL_ID`，然后：

```bash
docker compose up -d
```

## 模型热重载（无需重启）
```bash
curl -X POST http://localhost:12302/admin/reload \
  -H "Content-Type: application/json" \
  -H "x-admin-token: change-me" \
  -d '{
    "model_id":"Qwen/Qwen3-Embedding-4B",
    "max_model_len":8192,
    "gpu_memory_utilization":0.8
  }'
```

## 多 GPU 与选卡说明
- `deploy.resources.reservations.devices.count` 控制容器可见的 GPU 数量
- `NVIDIA_VISIBLE_DEVICES` 控制暴露宿主机哪一张卡给容器
  - `"0"` -> 宿主机第 1 张卡
  - `"1"` -> 宿主机第 2 张卡
  - `"0,1"` -> 同时暴露两张卡
- 如果只暴露一张卡给容器，那么容器里的 `vLLM` 会把它当作内部的 `cuda:0` 使用，这是正常现象
- 如果你要做多卡并行，可通过 `VLLM_EXTRA_ARGS` 追加参数，例如：

```yaml
VLLM_EXTRA_ARGS: "--tensor-parallel-size 2"
```

## 常用环境变量
在 `docker-compose.yml` 的 `environment` 里可调：
- `MODEL_ID`：模型 ID，默认 `Qwen/Qwen3-Embedding-4B`
- `NVIDIA_VISIBLE_DEVICES`：选卡
- `PORT`：外层 FastAPI 端口，默认 `12302`
- `VLLM_HOST` / `VLLM_PORT`：容器内 vLLM 监听地址，通常无需改
- `HF_HOME`：模型缓存目录
- `MAX_MODEL_LEN`：最大上下文长度，默认 `8192`（更适合 2080 Ti）
- `MAX_DIMENSIONS`：输出向量维度上限，默认 `2560`
- `GPU_MEMORY_UTILIZATION`：vLLM 显存利用率，默认 `0.80`
- `DEFAULT_QUERY_INSTRUCTION`：query 侧默认 instruction
- `ADMIN_TOKEN`：热重载接口鉴权
- `VLLM_EXTRA_ARGS`：透传额外 vLLM 参数

## 编码格式说明
- `encoding_format=float` 是当前保证可用的路径
- 其它编码格式会原样透传给底层 `vLLM`，是否支持取决于底层实现

## 测试
```bash
.venv/bin/python -m pytest --capture=no
```

## License
MIT License
