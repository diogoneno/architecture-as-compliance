"""aac.run: run identity (config, hashing, clock, seeds, env, sealing, index).

Rule: no module under aac.run imports torch at module level. Everything imports
aac.run, so report and pack code must stay CPU-only and fast to import.
"""
