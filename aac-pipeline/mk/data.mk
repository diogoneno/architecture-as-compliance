DATA_PY = $(VENV)/bin/python
TURING_LISTING ?=
FETCH_FROM_TURING = incrediblae/FC/train.tsv incrediblae/FC/dev.tsv incrediblae/FC/BERT-512.pth \
 incrediblae/RD/train.tsv incrediblae/RD/dev.tsv
CKPT_REL = averitec/pretrained_models/bert_veracity.ckpt
CKPT_URL = https://huggingface.co/chenxwh/AVeriTeC/resolve/2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031/pretrained_models/bert_veracity.ckpt
BENGIO_REQUIRED = averitec/data/train.json averitec/data/dev.json \
 incrediblae/FC/attack.tsv incrediblae/FC/train.tsv incrediblae/FC/dev.tsv incrediblae/FC/BERT-512.pth \
 incrediblae/RD/attack.tsv incrediblae/RD/train.tsv incrediblae/RD/dev.tsv incrediblae/RD/BERT-512.pth \
 $(CKPT_REL)

.PHONY: data-verify data-fetch ingest splits

data-verify:
	@set -e; L="$(TURING_LISTING)"; \
	if [ -z "$$L" ]; then L=$$(mktemp); trap 'rm -f "$$L"' EXIT; scripts/raw_hashes_remote.sh > "$$L"; fi; \
	$(DATA_PY) -m aac.data.verify --host gpu-host --listing "$$L" --require-all-expected; \
	$(DATA_PY) -m aac.data.verify --host cpu-host --extra --require $(BENGIO_REQUIRED)

data-fetch:
	@set -e; eval "$$($(AAC) dispatch env)"; \
	for rel in $(FETCH_FROM_TURING); do \
 mkdir -p "data/raw/$$(dirname "$$rel")"; \
 rsync -t --checksum -e "ssh $$AAC_SSH_OPTS" "$$AAC_REMOTE_HOST:$$AAC_RAW/$$rel" "data/raw/$$rel.part"; \
 $(DATA_PY) -m aac.data.verify --promote "data/raw/$$rel.part"; \
	done; \
	if [ -f "data/raw/$(CKPT_REL)" ]; then \
 echo "data-fetch: data/raw/$(CKPT_REL) present (hash-checked by make data-verify)"; \
	else \
 grep -qE "^[0-9a-f]{64} $(CKPT_REL)$$" configs/data/expected-extra.sha256 || \
 { echo "data-fetch: FAIL no expected sha256 for $(CKPT_REL) in configs/data/expected-extra.sha256 (re-read the HF API lfs.oid first)" >&2; exit 1; }; \
 mkdir -p "data/raw/$$(dirname "$(CKPT_REL)")"; \
 curl --fail -L -sS -o "data/raw/$(CKPT_REL).part" "$(CKPT_URL)" || { rm -f "data/raw/$(CKPT_REL).part"; exit 1; }; \
 $(DATA_PY) -m aac.data.verify --promote "data/raw/$(CKPT_REL).part"; \
	fi

ingest:
	$(DATA_PY) -m aac.data.averitec

splits:
	$(DATA_PY) -m aac.data.splits
