"""aac.attacks: the R3 attack side (filled by and Phase 3).

architectural isolation constraint rule : aac.attacks never imports the components package, directly or
transitively. The attack environment reaches the victim over localhost HTTP and the
query budget is counted in that victim wrapper (the attack-ledger `queries_used`
field), never inside the attack library. 's import-linter contract binds
this package as its source module.
"""
