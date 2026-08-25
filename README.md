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

### Lazy multiscale pyramids

The multiscale experiment reuses the existing level-5, level-4, and level-3
Virchow2 embedding artifacts. It keeps only complete aligned pyramids containing
1 coarse, 4 intermediate, and 16 fine tiles; no tiling or embedding job is run.

```bash
uv run -m ml.train +experiment=ml/train_mil_embeddings_multiscale
```

Tile extent is a validated Hydra hyperparameter rather than a hardcoded value.
When selecting a pyramid data card whose three level cards were tiled at 256
pixels, override it together with that card:

```bash
uv run -m ml.train +experiment=ml/train_mil_embeddings_multiscale \
  data/pyramid=my_256px_pyramid pyramid.tile_extent=256
```
