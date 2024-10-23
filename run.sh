#!/bin/bash
python -m histopipe.test +experiment=MMCI/test/test_con.yaml user=matejg
python -m histopipe.test +experiment=MMCI/test/test_mus.yaml user=matejg
python -m histopipe.test +experiment=MMCI/test/test_fat.yaml user=matejg
python -m histopipe.test +experiment=MMCI/test/test_ner.yaml user=matejg
python -m histopipe.test +experiment=MMCI/test/test_epi.yaml user=matejg
