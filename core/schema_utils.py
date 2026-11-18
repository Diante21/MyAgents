"""Turn a Pydantic model into a JSON schema the Claude API accepts for structured outputs.

Structured outputs (`output_config.format`) constrain the model's final answer to valid
JSON matching a schema. The API wants "closed" objects: every object must set
`additionalProperties: false` and list every property in `required`. Pydantic's
default schema doesn't do that and uses `$ref`/`$defs` for nested models, so we
inline the refs and close every object.
"""

from typing import Any

from pydantic import BaseModel


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    return _close(schema, defs)


def _close(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_close(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    # Inline references to nested models: {"$ref": "#/$defs/Foo"} -> Foo's schema
    if "$ref" in node:
        name = node["$ref"].split("/")[-1]
        return _close(dict(defs[name]), defs)

    out = {key: _close(value, defs) for key, value in node.items() if key != "title"}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out
