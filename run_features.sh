#!/bin/bash
python -m histopipe.test +experiment=TCGA/concepts/features/features_con.yaml user=matejg
<<<<<<< HEAD
python -m histopipe.test +experiment=TCGA/concepts/features/features_mus.yaml user=matejg
=======
#python -m histopipe.test +experiment=TCGA/concepts/features/features_mus.yaml user=matejg
>>>>>>> 5e78742 (feat: discovery configs)
python -m histopipe.test +experiment=TCGA/concepts/features/features_fat.yaml user=matejg
python -m histopipe.test +experiment=TCGA/concepts/features/features_ner.yaml user=matejg
python -m histopipe.test +experiment=TCGA/concepts/features/features_epi.yaml user=matejg
