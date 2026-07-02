#!/usr/bin/env python3
"""OpenAI-compatible proxy that parses Ornith/Qwen XML tool calls.

Ornith's model card says the model emits Qwen-style <tool_call> blocks and
expects the serving runtime to surface them as OpenAI `tool_calls`. llama.cpp
usually does this, but can leave tool XML in `reasoning_content`. This proxy
normalizes those missed cases for SWE-agent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from typing import Any

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^\n>]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([^\n>]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)


def _unframe_value(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def _tool_param_schemas(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        params = function.get("parameters") or {}
        props = params.get("properties") or {}
        if isinstance(name, str) and isinstance(props, dict):
            schemas[name] = {k: v for k, v in props.items() if isinstance(v, dict)}
    return schemas


def _schema_type(schema: dict[str, Any] | None) -> str | None:
    if not schema:
        return None
    typ = schema.get("type")
    if isinstance(typ, list):
        return next((x for x in typ if x != "null"), None)
    return typ if isinstance(typ, str) else None


def _coerce_value(value: str, schema: dict[str, Any] | None) -> Any:
    typ = _schema_type(schema)
    stripped = value.strip()

    if typ == "string":
        return value
    if typ in {"array", "object"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    if typ == "integer":
        try:
            return int(stripped)
        except ValueError:
            return value
    if typ == "number":
        try:
            return float(stripped)
        except ValueError:
            return value
    if typ == "boolean":
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value

    if stripped[:1] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def _parse_tool_calls(text: str, schemas: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        param_schemas = schemas.get(name, {})
        arguments: dict[str, Any] = {}
        for param_match in PARAM_RE.finditer(body):
            param_name = param_match.group(1).strip()
            raw_value = _unframe_value(param_match.group(2))
            arguments[param_name] = _coerce_value(raw_value, param_schemas.get(param_name))
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    return calls


def _remove_tool_xml(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def normalize_chat_response(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    schemas = _tool_param_schemas(payload)
    converted = 0

    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        if message.get("tool_calls"):
            continue

        sources = []
        content = message.get("content")
        if isinstance(content, str):
            sources.append(("content", content))
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            sources.append(("reasoning_content", reasoning))
        provider_fields = message.get("provider_specific_fields") or {}
        provider_reasoning = provider_fields.get("reasoning_content")
        if isinstance(provider_reasoning, str):
            sources.append(("provider_reasoning_content", provider_reasoning))

        for source_name, source_text in sources:
            calls = _parse_tool_calls(source_text, schemas)
            if not calls:
                continue

            message["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"
            converted += len(calls)

            if source_name == "content":
                message["content"] = _remove_tool_xml(source_text)
            elif source_name == "reasoning_content":
                message["reasoning_content"] = _remove_tool_xml(source_text)
                if isinstance(provider_fields, dict) and "reasoning_content" in provider_fields:
                    provider_fields["reasoning_content"] = message["reasoning_content"]
            elif source_name == "provider_reasoning_content":
                provider_fields["reasoning_content"] = _remove_tool_xml(source_text)
                message["reasoning_content"] = provider_fields["reasoning_content"]
            break

    if converted:
        print(f"ornith_tool_proxy: converted {converted} XML tool call(s)", file=sys.stderr, flush=True)
    return data


def create_app(upstream: str) -> FastAPI:
    app = FastAPI()
    upstream = upstream.rstrip("/")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "upstream": upstream}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        url = f"{upstream}/{path}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        body = await request.body()

        if request.method == "POST" and path == "v1/chat/completions":
            payload = json.loads(body.decode("utf-8")) if body else {}
            if payload.get("stream"):
                upstream_response = requests.post(
                    url, headers=headers, json=payload, stream=True, timeout=None
                )
                return StreamingResponse(
                    upstream_response.iter_content(chunk_size=None),
                    status_code=upstream_response.status_code,
                    media_type=upstream_response.headers.get("content-type"),
                )
            upstream_response = requests.post(url, headers=headers, json=payload, timeout=None)
            if upstream_response.headers.get("content-type", "").startswith("application/json"):
                data = upstream_response.json()
                if upstream_response.ok:
                    data = normalize_chat_response(data, payload)
                return JSONResponse(data, status_code=upstream_response.status_code)
            return Response(
                upstream_response.content,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type"),
            )

        upstream_response = requests.request(
            request.method, url, headers=headers, data=body or None, timeout=None
        )
        return Response(
            upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--upstream", default="http://127.0.0.1:8091")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.upstream), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
