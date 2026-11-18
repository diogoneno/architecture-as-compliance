"""The chunk table of the untargeted RD BERT-Attack (Table 4.1 row 4).

Why chunks: the single attack.py process over all 415 RD rows grows its memory with the items processed (attempt 2,
run alone: RSS 14.9 GB after 63 min, about 27 items, roughly linear) and the analyst stopped it. Each chunk is one
fresh attack.py process (BODEGA's script unchanged) over attack.tsv rows [start, stop) and one sealed BASE run
(external.row_range); Table 4.1 row 4 pools the chunks (aac.reports.table_4_1).

Chunk size: 14.9 GB / 27 items = 0.55 GB per item, read as if the whole RSS grew with the items (conservative: it
includes the model and scorer baseline). 25 rows -> 25 x 0.55 = 13.8 GB, about 2 GB under the 16 GB a process may use
(cpu-host has 30 GB RAM; swap is not planned on). 415 = 16 x 25 + 15 -> 17 chunks.

One table for everything that needs it: scripts/make_rd_chunk_configs.py writes the configs from it, mk/baselines.mk
runs the configs it wrote, table_4_1 reads exactly CHUNK_NAMES (a missing chunk is MissingRun, never a partial pool)."""
from __future__ import annotations

N_ROWS = 415            # data rows of data/raw/incrediblae/RD/attack.tsv (no header; wc -l = 415), = rd-attack.items.json
CHUNK_SIZE = 25
BASE_NAME = "bodega-rd-bertattack-untargeted"


def chunks(n_rows: int = N_ROWS, size: int = CHUNK_SIZE) -> tuple[tuple[str, int, int], ...]:
    """((chunk id, start, stop), ...): c01, c02, ... over [0, n_rows), stop exclusive, the last chunk shorter."""
    width = max(2, len(str(-(-n_rows // size))))
    return tuple((f"c{i + 1:0{width}d}", s, min(s + size, n_rows)) for i, s in enumerate(range(0, n_rows, size)))


CHUNKS = chunks()


def config_name(chunk_id: str) -> str:
    return f"{BASE_NAME}-{chunk_id}"


CHUNK_NAMES = tuple(config_name(c) for c, _s, _e in CHUNKS)
