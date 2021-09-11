# METHOD FOR PREDICTING TYROSINE KINASE INHIBITOR (TKI) RESISTANCE IN CCRCC PATIENTS

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Demo](#demo)
- [Results](#results)

# Overview
To construct a classifier to predict the drug response of patients. We used proteome data of response patients as baseline. We build a reference interval (RI) with the distribution of response protein abundance, which could be used for anomaly detection if the protein abundance exceeds the upper limit of the RI. For any query sample, an outlier is defined as a protein whose expression level is higher than the upper limit of the RI, and the detection of multiple outliers would further increase the statistical confidence for patients with non-response. To this end, we constructed proteomic RIs derived from 29 response samples. The upper limit of the RI is defined as P75 + 3*(P75 – P25), where P75 and P25 are the 75th percentile and 25th percentile of the protein abundance in the response groups, respectively. We trained a linear classifier to classify the patient with different drug response using significance of drug response outliers. 


# Contents

- [code](./code): Python code.
- [demo](./demo): Demo input and output.



# System Requirements

## Hardware Requirements

The script requires only a standard computer.
## Software Requirements

### OS Requirements

The package development version is tested on both *Linux* and *Windows* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 16.04  
Mac OSX:  
Windows:  

Before using the script, users should have `Python` version 3.0.0 or higher.

```
pip install numpy
pip install scipy
pip install pandas
pip install scikit-learn
```

# Demo

```
python prediction.py 
```


# Results



