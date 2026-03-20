import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from embedding_service import (
    DEFAULT_QUERY_INSTRUCTION,
    InputValidationError,
    BackendProxyError,
    BackendUnavailableError,
    embed_texts,
    get_health_snapshot,
)

def build_health_resource_content() -> str:
    return json.dumps(get_health_snapshot(), ensure_ascii=False, indent=2)


def build_usage_resource_content() -> str:
    lines = [
        "# Qwen3-Embedding MCP Usage",
        "",
        "Use `embed_text` to generate embeddings through the wrapper service.",
        "",
        "Arguments:",
        "- `texts` (required): one string or a list of strings",
        "- `input_type` (optional): `query` or `document`",
        "- `instruction` (optional): query-side instruction for Qwen retrieval formatting",
        "- `dimensions` (optional): output dimensions between 32 and 4096",
        "",
        "Notes:",
        "- Query embeddings are wrapped as `Instruct: ...\\nQuery:...` before being sent to Qwen.",
        f"- If `instruction` is omitted for queries, the service uses the default: `{DEFAULT_QUERY_INSTRUCTION}`",
    ]
    return "\n".join(lines)


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("Qwen3-Embedding", stateless_http=True, json_response=True)
    mcp.settings.streamable_http_path = "/"

    @mcp.tool()
    async def embed_text(
        texts: str | list[str],
        input_type: Optional[str] = None,
        instruction: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> dict:
        """Generate text embeddings through the local vLLM-backed service."""
        try:
            return await embed_texts(
                texts=texts,
                input_type=input_type,
                instruction=instruction,
                dimensions=dimensions,
            )
        except InputValidationError as exc:
            raise RuntimeError(f"invalid_input: {exc}") from exc
        except BackendUnavailableError as exc:
            raise RuntimeError(f"backend_unavailable: {exc}") from exc
        except BackendProxyError as exc:
            if exc.payload is not None:
                raise RuntimeError(json.dumps(exc.payload, ensure_ascii=False)) from exc
            raise RuntimeError(f"backend_error: {exc}") from exc

    @mcp.resource("qwen3embedding://health")
    def qwen3embedding_health() -> str:
        """Expose read-only runtime status for MCP clients."""
        return build_health_resource_content()

    @mcp.resource("qwen3embedding://usage")
    def qwen3embedding_usage() -> str:
        """Describe how to call the embedding tool safely."""
        return build_usage_resource_content()

    @mcp.prompt()
    def retrieval_embedding_workflow() -> str:
        """Guide retrieval-oriented clients to use the tool correctly."""
        return (
            "When embedding search queries, call `embed_text` with `input_type=query`. "
            "Pass a task-specific English `instruction` whenever possible. "
            "When embedding passages or chunks for indexing, use `input_type=document` and do not add the instruction. "
            "Keep query and document embeddings in separate requests."
        )

    return mcp


mcp = create_mcp_server()
