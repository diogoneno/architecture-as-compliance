"""Draft 2020-12 validator over contracts/decision-log.schema.json with a strict
format checker (uuid, date-time) that does not depend on optional packages."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from aac.run.paths import root

_DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$")


class SchemaViolation(Exception):
    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def schema_path() -> Path:
    return root() / "contracts" / "decision-log.schema.json"


def load_schema(path: Path | None = None) -> dict:
    with open(path or schema_path(), encoding="utf-8") as fh:
        return json.load(fh)


def format_checker() -> FormatChecker:
    checker = FormatChecker(formats=())

    @checker.checks("uuid")
    def _uuid(value: object) -> bool:
        if not isinstance(value, str):
            return True
        try:
            return str(uuid.UUID(value)) == value
        except ValueError:
            return False

    @checker.checks("date-time")
    def _dt(value: object) -> bool:
        if not isinstance(value, str):
            return True
        return bool(_DATE_TIME.match(value))

    return checker


def load_validator(path: Path | None = None) -> Draft202012Validator:
    schema = load_schema(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=format_checker())


def errors_for(validator: Draft202012Validator, record: dict) -> list[str]:
    return sorted(f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in validator.iter_errors(record))


def check(validator: Draft202012Validator, record: dict) -> None:
    errs = errors_for(validator, record)
    if errs:
        raise SchemaViolation(f"record violates decision-log schema: {errs[0]}", errs)
