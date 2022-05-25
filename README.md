# A Psychological Taxonomy of Anti-Vaccination Arguments: Systematic Literature Review and Text Modeling -- Text Classification

Source code for the text classification experiments from A Psychological Taxonomy of Anti-Vaccination Arguments: Systematic Literature Review and Text Modeling (https://osf.io/e4yp6/)

Contact person: Luke Bates, bates@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
* `main.py` -- code file that uses the other code files
* `data/full_multilabel` -- 11 multilabel attitude root data from the abstract level of argumentation from Study 1 and from the Study 2 fact checks.
* `data/full_single_label` -- Single-label, specific (meta) level from study 1 and dominate root fact check data from Study 2.
* `data/mutlilabel_collapse` -- 7 attitude root data from the specific level of argumentation from Study 1 and from the Study 2 fact checks.
* `data/single_collapse` -- single label 7 attitude root data from the specific level of argumentation from Study 1 and from the Study 2 fact checks.
* `output` -- where modelling results in the form of json files are written
* `written_models_full` -- where transformers are written to disk for pretraining and two-step finetuning with all 11 roots
* `written_models_collapse` -- where transformers are written to disk for pretraining and two-step finetuning with 7 roots
* `umap_plots` -- source code and data used for the umap plots from the paper. Please see the README.md file there.


## Requirements
Our results were computed in Python 3.6.8 with a 40gb amphere A100 GPU. Note that files will be written to disk if the code is run.


## Installation
To setup, please follow the instructions below.
```
python -m venv mvenv
source mvenv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
````
 
Then, you can run the code with `python main.py`.


### Expected results
Once finished, the results will be written to json files in the "output"  folder. The "mean_f1_mac" field is the macro F1 metric reported in the paper.

* "full" refers to the 11 attitude roots while "collapse" refers to the 7 attitude root scenario.
* "test" refers evalutation results, which are reported in the paper. "train" refers to model optimization results.
* "transformer" refers to RoBERTa-base from the paper.
* "setfit" refers to the SetFit from the paper.
* "transformer_pretrain" is modelling done on Study 1 data and setups the two-step finetuning results. The results here are not reported in the paper.
* "zero" refers to the "zero-shot" results
* "transformer_baseline" is standard fine-tuning results with RoBERTa-base on Study 2.
* "use_ft_transformer" is the two-step fine-tuning with RoBERTa-base.
* "st_baseline" is the Sentence Transformer and logistic regression.
* "setfit_baseline" is standard fine-tuning with SetFit.
* "setfit_pretrain" is two-step fine-tuning with SetFit.

If you wish to see a summary of the results reported in the paper you can use `python summarize.py`.
