# Qwen3 Embedding：自托管 Embedding 推理服务

把 `Qwen/Qwen3-Embedding-8B` 封装成一个可自托管的推理服务：对外提供 OpenAI 兼容的 `POST /v1/embeddings`、HTTP MCP Server、内置调试页面，以及健康检查和模型热重载，便于在内网或私有环境中快速接入。

## 功能
- OpenAI 兼容 Embeddings API：`POST /v1/embeddings`
- Qwen 扩展字段：`input_type=query|document`、`instruction`
- MCP Server：`POST/GET /mcp`
- 内置 Web UI：`GET /`
- 健康检查：`GET /health`
- 交互式接口文档：`GET /docs` 与 `GET /redoc`
- 模型热重载：`POST /admin/reload`（`ADMIN_TOKEN` 保护）
- 模型自动下载与缓存：挂载 `./models` 到容器 `/models`

## 架构
- 外层服务：FastAPI
- 推理后端：同容器内拉起的 `vLLM` 子进程
- 默认模型：`Qwen/Qwen3-Embedding-8B`
- 默认端口：外部 `12301`，内部 vLLM `8001`

## 快速开始
```bash
docker compose up -d --build
```

打开：
- Web UI：http://localhost:12301/
- Swagger：http://localhost:12301/docs
- ReDoc：http://localhost:12301/redoc
- MCP HTTP：http://localhost:12301/mcp
- 健康检查：http://localhost:12301/health

## OpenAI SDK 示例
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12301/v1",
    api_key="dummy",
)

response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
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
curl http://localhost:12301/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-8B",
    "input": [
      "What is the capital of China?",
      "The capital of China is Beijing."
    ],
    "input_type": "query",
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "dimensions": 1024
  }'
```

## API 一览
- `POST /v1/embeddings`
  - 标准字段：`input`、`model`、`dimensions`、`encoding_format`、`user`
  - 扩展字段：`input_type`、`instruction`
- `GET /health`
- `POST /admin/reload`
- `POST /mcp` / `GET /mcp`

## Qwen 扩展说明
- `input_type=query` 时，服务会把每条输入包装为：
  - `Instruct: {instruction}\nQuery:{text}`
- 若未传 `instruction`，使用默认英文检索 instruction。
- `input_type=document` 或未传 `input_type` 时，不自动改写输入。
- 单次请求内的所有输入共享同一个 `input_type` 和 `instruction`。

## 热重载
```bash
curl -X POST http://localhost:12301/admin/reload \
  -H "Content-Type: application/json" \
  -H "x-admin-token: change-me" \
  -d '{
    "model_id": "Qwen/Qwen3-Embedding-8B",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.9
  }'
```

## 代理与模型缓存
如需走代理访问 Hugging Face，可在 `.env` 中设置：

```bash
HTTP_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1
```

模型缓存目录默认为容器内 `/models`，请通过 volume 挂载到宿主机持久化。

## 资源说明
- `Qwen/Qwen3-Embedding-8B` 为 8B 级别 embedding 模型，v1 默认面向 NVIDIA GPU。
- `dimensions` 支持 `32-4096`，利用模型的 Matryoshka 能力压缩向量长度。
- 生产部署建议关注 GPU 显存、`MAX_MODEL_LEN` 和 `GPU_MEMORY_UTILIZATION` 的组合。

## 测试
```bash
pytest
```
