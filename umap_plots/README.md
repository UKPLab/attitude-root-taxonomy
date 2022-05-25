# A Psychological Taxonomy of Anti-Vaccination Arguments: Systematic Literature Review and Text Modeling -- UMAP plots

Source code for the UMAP plots from A Psychological Taxonomy of Anti-Vaccination Arguments: Systematic Literature Review and Text Modeling (https://osf.io/e4yp6/)

Contact person: Luke Bates, bates@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
* `umap_plots.ipynb` -- code file
* `data/abstract` -- 11 (full) and 7 (collapsed) single-label attitude root data from the abstract level of argumentation from Study 1.
* `data/full_single_label` -- Single-label, specific level from study 1 and single-label fact check data from Study 2.
* `output` -- plot file output folder

## Requirements
The plots from the paper were made in Python 3.8.10 on an NVIDIA GeForce RTX 3060 GPU. Note that UMAP requires Python 3.7+.


## Installation
To setup, please follow the instructions below.

````
python -m venv mvenv
source mvenv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
````
 
Then, you can run the code with in a Jupyter notebook for ease of viewing: `jupyter notebook`

### Expected results
Plot files will be written to the "output" directory. You can also view them in the notebook.

