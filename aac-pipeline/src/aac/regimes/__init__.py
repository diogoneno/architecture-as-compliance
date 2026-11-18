"""aac.regimes: one driver per regime (BASE, R1-R4), all on the single Pipeline.infer path.
Drivers import Pipeline, aac.data providers and aac.attacks only, never the components package (architectural isolation constraint)."""
