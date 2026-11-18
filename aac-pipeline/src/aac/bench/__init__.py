"""Day-1 benchmarks : aac.bench.retrieval seals a BENCH run (BGE-M3 and bge-reranker-v2-m3
throughput on real store passages, bit-identity within and across processes, batch-1 latency per stage);
aac.bench.report writes reports/bench-bge-m3.csv from it, with the C1 determinism double run and the
recomputed Table F.2 indexing projection. torch / sentence-transformers are imported inside functions."""
