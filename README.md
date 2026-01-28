# Code for "TimeMAC: Collaborative Multi-Agent Forecasting for Hierarchical Time Series" 

"TimeMAC: Collaborative Multi-Agent Forecasting for Hierarchical Time Series" is under reviewing at ICML 2026.

## What is TimeMAC?

## Method Overview

![framework](figures/main_TimeMAC.pdf)

## Setup

All experiments run in Python 3.12 environment. You can install the dependency libraries as follows:

```
pip install -r requirements.txt
```

## Run the Code

All the methods compared in the paper can be run as follows. Our method is denoted as "TimeMAC".

```
python run_exp.py
```

where dataset is one of `{Tourismsmall, Tourismlarge, Traffic, Labour, Wiki, Amazon}`.



