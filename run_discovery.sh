#!/bin/bash
python -m histopipe.test +experiment=TCGA/concepts/discovery/discovery_con.yaml user=matejg
#python -m histopipe.test +experiment=TCGA/concepts/discovery/discovery_mus.yaml user=matejg
python -m histopipe.test +experiment=TCGA/concepts/discovery/discovery_fat.yaml user=matejg
python -m histopipe.test +experiment=TCGA/concepts/discovery/discovery_ner.yaml user=matejg
python -m histopipe.test +experiment=TCGA/concepts/discovery/discovery_epi.yaml user=matejg
