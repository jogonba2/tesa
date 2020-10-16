# TESA: A Task in Entity Semantic Aggregation for AbstractiveSummarization

## Code

TODO

## Data

Our dataset is available through the Linguistic Data Consortium, at the following link:

## Installation

/toolbox/ubuntu_environment.yml contains the environment I used in my laptop. Also, it is required to install Fairseq and pywikibot as libraries:

```console
git clone https://github.com/jogonba2/tesa.git
conda env create -f ./toolbox/ubuntu_environment.yml
pip install --editable ./fairseq/setup.py
pip install --editable ./pywikibot/setup.py
```
