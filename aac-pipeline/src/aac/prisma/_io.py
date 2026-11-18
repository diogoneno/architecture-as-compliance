"""Confined, atomic file I/O for aac.prisma (Security V12).

Every aac.prisma writer calls assert_confined before a byte is written: the only accepted targets are paths
under <appendix>/screening/ and exactly <appendix>/reference-verification-YYYY-MM-DD.csv. the analyst's
run folders (run-*), _superseded/, chapters/, tools/, `..` escapes and symlinks that resolve elsewhere raise
OutsideScreening. Writes go to a `.tmp` sibling and os.replace; the tmp file is unlinked on any exception
(the aac.data.averitec._write_atomic shape, in UTF-8 because many titles are non-ASCII).
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from aac.run.paths import root as project_root

SCREENING = "screening"
LOCK_NAME = ".workbook.lock"
REFCHECK_RE = re.compile(r"^reference-verification-\d{4}-\d{2}-\d{2}\.csv$")


class OutsideScreening(RuntimeError):
    """A write target outside appendix-c/screening/ (and not the one reference-verification CSV): refused."""


def default_appendix_dir() -> Path:
    """<dissertation root>/appendix-c: the sibling of aac-pipeline/ (AAC_ROOT moves it with the root)."""
    return project_root().parent / "appendix-c"


def screening_dir(appendix: Path) -> Path:
    return Path(appendix) / SCREENING


def assert_confined(path: Path, appendix: Path) -> Path:
    """Return the resolved target when it may be written, else raise OutsideScreening naming the path."""
    app = Path(appendix).resolve()
    scr = app / SCREENING
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    resolved = target.resolve()  # follows symlinks in every existing component
    if resolved != scr and resolved.is_relative_to(scr):
        return resolved
    if resolved.parent == app and REFCHECK_RE.match(resolved.name):
        return resolved
    raise OutsideScreening(
        f"refusing to write {path}: aac.prisma writes only under {scr}/ "
        f"or {app}/reference-verification-YYYY-MM-DD.csv (resolved: {resolved})"
    )


def write_text_atomic(path: Path, text: str, appendix: Path, encoding: str = "utf-8") -> Path:
    """Confined tmp + os.replace write; a failed write leaves neither the target nor the tmp file."""
    target = assert_confined(path, appendix)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def csv_text(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """The exact text write_csv_atomic writes: csv.writer with LF line ends (the repo writer convention)."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(header))
    for r in rows:
        w.writerow(list(r))
    return buf.getvalue()


def write_csv_atomic(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]], appendix: Path) -> Path:
    assert_confined(path, appendix)  # before the rows are even rendered
    return write_text_atomic(path, csv_text(header, rows), appendix)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """(header, rows as dicts); a BOM is dropped and CRLF or LF line ends are both read."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


@contextlib.contextmanager
def workbook_lock(appendix: Path, shared: bool) -> Iterator[None]:
    """fcntl.flock on <appendix>/screening/.workbook.lock: LOCK_SH for readers, LOCK_EX for writers.

 A reader on an appendix that has no screening/ folder yet takes no lock (there is nothing to read)."""
    scr = screening_dir(appendix)
    if shared and not scr.is_dir():
        yield
        return
    lock = assert_confined(scr / LOCK_NAME, appendix)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a", encoding="ascii") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
