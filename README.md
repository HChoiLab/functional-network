# Stimulus type shapes the topology of cellular functional networks in mouse visual cortex

[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://www.nature.com/articles/s41467-024-49704-0)



## Introduction

This repository contains the code and data to reproduce the results presented in the paper "Stimulus type shapes the topology of cellular functional networks in mouse visual cortex."

To generate figures in the manuscript, simply run the corresponding jupyter notebooks where functions can be found in library.py.

To run signed motif analysis, use signed_motif_detection.py where example usage can be found in signed_motif_detection_example.ipynb.

To run jitter-corrected CCG, use ccg_library.py where example usage can be found in ccg_example_usage.ipynb.

## Abstract

On the timescale of sensory processing, neuronal networks have relatively fixed anatomical connectivity, while functional interactions between neurons can vary depending on the ongoing activity of the neurons within the network. We thus hypothesized that different types of stimuli could lead those networks to display stimulus-dependent functional connectivity patterns. To test this hypothesis, we analyzed single-cell resolution electrophysiological data from the Allen Institute, with simultaneous recordings of stimulus-evoked activity from neurons across 6 different regions of mouse visual cortex. Comparing the functional connectivity patterns during different stimulus types, we made several nontrivial observations: (1) while the frequencies of different functional motifs were preserved across stimuli, the identities of the neurons within those motifs changed; (2) the degree to which functional modules are contained within a single brain region increases with stimulus complexity. Altogether, our work reveals unexpected stimulus-dependence to the way groups of neurons interact to process incoming sensory information.

## Paper Link

The full paper can be accessed [here](https://www.nature.com/articles/s41467-024-49704-0).

## Python dependencies
```
numpy==1.23.5
scipy==1.9.3
pandas==1.5.2
statsmodels==0.13.2
networkx==2.8.8
python-louvain==0.16
netgraph==4.10.2
scikit-learn==1.1.3
upsetplot==0.8.0
```
## Citation

If you use this code or data in your research, please cite our paper:
```bibtex
@article{tang2024stimulus,
  title={Stimulus type shapes the topology of cellular functional networks in mouse visual cortex},
  author={Tang, Disheng and Zylberberg, Joel and Jia, Xiaoxuan and Choi, Hannah},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={5753},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
