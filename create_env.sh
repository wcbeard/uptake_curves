set -euo pipefail

conda env create -f environment.yml

source activate uptake_curves

jupyter labextension install jupyterlab_vim @jupyterlab/toc @ryantam626/jupyterlab_code_formatter

cd notebooks
jupytext --to ipynb templ.py
cd ..
