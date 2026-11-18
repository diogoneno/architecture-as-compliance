"""External baselines (Table 4.1 rows 1-5): third-party models reproduced with the third
parties' own code and checkpoints on the CPU host's CPU, sealed as BASE runs with record_count 0 that hold numbers and
row-indexed outcomes, never text.

 external ExternalConfig runner (`python -m aac.baselines.external run`), private staging, sealing
 bodega BODEGA's unchanged runs/test_victims.py and runs/attack.py in env-attack (scripts/attack_env.sh)
 averitec_gold the AVeriTeC baseline veracity checkpoint on the dev gold question-answer pairs (env-main)"""
