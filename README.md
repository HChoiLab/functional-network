# Stimulus-dependent functional network topology in mouse visual cortex

[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://www.biorxiv.org/content/10.1101/2023.07.03.547364v1)



## Introduction

This repository contains the code and data to reproduce the results presented in the paper "Stimulus-dependent functional network topology in mouse visual cortex."

To generate figures in the manuscript, simply run the corresponding jupyter notebooks where functions can be found in library.py.

## Abstract

Information is processed by networks of neurons in the brain. On the timescale of sensory processing, those neuronal networks have relatively fixed anatomical connectivity, while functional connectivity, which defines the interactions between neurons, can vary depending on the ongoing activity of the neurons within the network. We thus hypothesized that different types of stimuli, which drive different neuronal activities in the network, could lead those networks to display stimulus-dependent functional connectivity patterns. To test this hypothesis, we analyzed electrophysiological data from the Allen Brain Observatory, which utilized Neuropixels probes to simultaneously record stimulus-evoked activity from hundreds of neurons across 6 different regions of mouse visual cortex. The recordings had single-cell resolution and high temporal fidelity, enabling us to determine fine-scale functional connectivity. Comparing the functional connectivity patterns observed when different stimuli were presented to the mice, we made several nontrivial observations. First, while the frequencies of different connectivity motifs (i.e., the patterns of connectivity between triplets of neurons) were preserved across stimuli, the identities of the neurons within those motifs changed. This means that functional connectivity dynamically changes along with the input stimulus, but does so in a way that preserves the motif frequencies. Secondly, we found that the degree to which functional modules are contained within a single brain region (as opposed to being distributed between regions) increases with increasing stimulus complexity. This suggests a mechanism for how the brain could dynamically alter its computations based on its inputs. Altogether, our work reveals unexpected stimulus-dependence to the way groups of neurons interact to process incoming sensory information.

## Paper Link

The full paper can be accessed [here](https://www.biorxiv.org/content/10.1101/2023.07.03.547364v1).

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
@article{tang2023stimulus,
  title={Stimulus-dependent functional network topology in mouse visual cortex},
  author={Tang, Disheng and Zylberberg, Joel and Jia, Xiaoxuan and Choi, Hannah},
  journal={bioRxiv},
  pages={2023--07},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
