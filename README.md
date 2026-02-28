**Predicting future Service Level Agreement (SLA) violations using online learning**

## Repository structure

```text
.
├── 1_LITERATURE/            # papers and references used for this project
├── LICENSE                  
├── README.md                
├── column_names.txt         # list of column names in the dataset
├── helpers.py               # helper functions for preprocessing
├── modelling.ipynb          # main notebook (experiments, results)
├── offline_cross_trace.py   # functions for offline cross-tarce
├── online_single_trace.py   # functions for online single trace
└── summary.docx             # papers summary

````


This repo implements an **early-warning system for SLA violations** in a video streaming setting. Using **server-side device metrics** (CPU, memory, disk, network, else) sampled each second, the target is to predict an upcoming client-side quality drop: given the last $W$ seconds of metrics, predict whether a drop of quality will occur within the next $H$ seconds, conditioned on the system being healthy now.

Methods used
- **Offline models** Logistic Regression and Random Forest evaluated on single trace and cross-trace
- **Online learning** Logistic regression and OAUE evaluated on single trace and cross-trace

