"""aac.serve: the R3 victim server (server half of resume clause of ).

One env-main process owns the R3 run, its decision log (one ShardWriter, one seq), the per-item query
budgets and the attack ledger. The OpenAttack runner (env-attack, CPython 3.10) reaches it only over
HTTP/1.1 on an AF_UNIX socket (victim protocol v1, attacks/runner/protocol.py). aac.serve may import
aac.pipeline.factory and aac.components like the CLI; aac.regimes and aac.attacks never import aac.serve."""
