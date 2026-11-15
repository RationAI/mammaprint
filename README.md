# MammaPrint

Michal Kišš, Matěj Pekár

## Getting started

### Instalation

Install [UV](https://docs.astral.sh/uv/) package manager and install all dependencies using the following command:
```bash
uv sync
```

### Preprocessing

Use scripts/preprocessing scripts.

### Modular multiscale MIL

The modular multiscale experiments reuse the existing level-5, level-4, and
level-3 Virchow2 embedding artifacts. They keep only complete aligned pyramids
containing 1 coarse, 4 intermediate, and 16 fine tiles; no tiling or embedding
job is run. Start with the parameter-free mean/concat/mean baseline:

```bash
uv run python -m ml.train +experiment=ml/multiscale/multiscale_mean_linear
```

See the [modular multiscale MIL configuration reference](docs/multiscale-mil-config.md)
for stage options, compatible combinations, named presets, and data-card
overrides.

Select a final mask dataset and level window with `data/pyramid`, for example:

```bash
uv run python -m ml.train +experiment=ml/multiscale/multiscale_transformer \
  data/pyramid=epithel_and_cancer_5_l5_l4_l3
```

All six final embedding families support L5/L4/L3 and L4/L3/L2; see the
configuration reference for the available selections.
