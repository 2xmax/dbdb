[![Build status](https://dev.azure.com/supatfs/dbdb/_apis/build/status/dbdb-Python%20package-CI)](https://dev.azure.com/supatfs/dbdb/_build/latest?definitionId=1) [!["THE SUSHI-WARE LICENSE"](https://img.shields.io/badge/license-SUSHI--WARE%F0%9F%8D%A3-blue.svg)](https://github.com/MakeNowJust/sushi-ware)

This repository provides source code for Maximov MA, Galukhin AV, and Gor GY, "Pore Size Distribution of Silica Colloidal Crystals from Nitrogen Adsorption Isotherms", Langmuir 2019 (accepted)

In short, it generates a kernel of adsorption and desorption isotherms using Frenkel-Halsey-Hill and Derjaguin-Broekhoff-de Boer theories and extracts the pore size distribution using Non-Negative Least Squares regression with Tikhonov (ridge) regularization and Generalized Cross-Validation.

![alt text](https://github.com/2xmax/dbdb_private/blob/docs/docs/TOC.png?raw=true "TOC")

Quick start
===========
To install the package without cloning the project, run ```pip install dbdb```.

To run examples from docs, clone the project ```git clone https://github.com/2xmax/dbdb.git && cd dbdb```, run ```pip install . && pip install -r requirements.docs.txt && jupyter notebook``` in terminal/cmd and open the notebooks in your browser. Python 3.5+ is required for the source code, 3.6+ is required for the notebooks with examples.

Description of notebooks:
 - [psd_n2.ipynb](https://nbviewer.jupyter.org/github/2xmax/dbdb/blob/master/docs/psd_n2.ipynb) - quick introduction based on the article data
- [maximov2019lang_figures.ipynb](https://nbviewer.jupyter.org/github/2xmax/dbdb/blob/master/docs/maximov2019lang_figures.ipynb) - reproduced pictures from the current article
- [galukhin2019lang_figures.ipynb](https://nbviewer.jupyter.org/github/2xmax/dbdb/blob/master/docs/galukhin2019lang_figures.ipynb) - reproduced pictures from the [previous article](https://doi.org/10.1021/acs.langmuir.8b03476) in series
- [psd_arc.psd](https://nbviewer.jupyter.org/github/2xmax/dbdb/blob/master/docs/psd_arc.ipynb) - application of FHH and DBdB theories on data from [Cychosz, Katie A., et al., Langmuir 28.34 (2012): 12647-12654.](https://doi.org/10.1021/la302362h)

Kernels
===============
Pre-calculated kernels for the article are available on the [releases page](https://github.com/2xmax/dbdb/releases). If you want to reparametrize it, go to docs/maximov2019lang_figures.ipynb and see how it was done.

BibTeX citation
===============
```
@article{maximov2019opals,
  title={Pore Size Distribution of Silica Colloidal Crystals from Nitrogen Adsorption Isotherms},
  author={Maximov, Max A and Galukhin, Andrey V and Gor, Gennady Y},
  journal={Langmuir},
  pages={accepted},
  year={2019},
  publisher={ACS Publications}
}
```
 
