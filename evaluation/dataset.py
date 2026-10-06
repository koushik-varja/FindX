from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_MODALITIES = {"text", "image", "multimodal"}


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    query: str
    language: str
    modality: str
    query_type: str
    relevant_document_ids: tuple[str, ...]
    graded_relevance: dict[str, int]
    notes: str = ""
    query_image_id: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GoldenQuery":
        relevant = tuple(str(value) for value in row.get("relevant_document_ids", []))
        grades = {str(key): int(value) for key, value in row.get("graded_relevance", {}).items()}
        if not grades:
            grades = {doc_id: 1 for doc_id in relevant}
        if not relevant:
            relevant = tuple(grades)
        item = cls(
            query_id=str(row["query_id"]),
            query=str(row.get("query", "")),
            language=str(row.get("language", "unknown")),
            modality=str(row["modality"]),
            query_type=str(row.get("query_type", "unknown")),
            relevant_document_ids=relevant,
            graded_relevance=grades,
            notes=str(row.get("notes", "")),
            query_image_id=row.get("query_image_id"),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not self.query_id:
            raise ValueError("query_id is required")
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality}")
        if self.modality == "text" and not self.query.strip():
            raise ValueError(f"text query {self.query_id} has no query text")
        if self.modality in {"image", "multimodal"} and not self.query_image_id:
            raise ValueError(f"{self.modality} query {self.query_id} needs query_image_id")
        if not self.graded_relevance:
            raise ValueError(f"query {self.query_id} has no relevance labels")


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    rows: list[GoldenQuery] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        item = GoldenQuery.from_dict(row)
        if item.query_id in seen:
            raise ValueError(f"duplicate query_id: {item.query_id}")
        seen.add(item.query_id)
        rows.append(item)
    return rows
