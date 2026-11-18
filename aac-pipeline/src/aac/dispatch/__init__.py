"""Dispatch to the GPU host : ssh/scp wrappers, the GPU and clean-tree guards,
rsync --checksum transport and the run status tail. the analyst's GPU rule lives here as code:
gpu-run needs BOTH the analyst-created sentinel AND zero compute apps on the card ."""
