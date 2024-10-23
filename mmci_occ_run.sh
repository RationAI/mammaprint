#!/bin/bash
python -m histopipe.test +experiment=MMCI/explain/occ_con.yaml user=matejg
python -m histopipe.test +experiment=MMCI/explain/occ_mus.yaml user=matejg
python -m histopipe.test +experiment=MMCI/explain/occ_fat.yaml user=matejg
python -m histopipe.test +experiment=MMCI/explain/occ_ner.yaml user=matejg
python -m histopipe.test +experiment=MMCI/explain/occ_epi.yaml user=matejg
