"""aac.stats: the ONLY source of CIs, p-values and effect sizes for every table .

bootstrap: percentile / paired-index / unpaired-index bootstrap keyed by table cell
tests: paired_t_dz, wilcoxon_descriptive (with floor p), mcnemar_exact, welch_t (+ welch_df)
holm: Holm within a specification family, input order preserved
families: the H1..H4 ids per regime from configs/stats.yaml
"""
