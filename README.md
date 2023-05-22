# Conformal Prediction with Temporal Quantile Adjustments

This is the code associated with "Conformal Prediction with Temporal Quantile Adjustments" (NeurIPS 2022).

To replicate the (`GEFCom-R`) experiments:
1. Train the base models: `python -m utils.main_experiments`
2. run `main.ipynb` for results

For other datasets, please download the corresponding data (see `data.preprocessing` for more details) and change the `__main__` section in `utils.main_experiments`.

## Demo
The [demo](https://github.com/zlin7/TQA/blob/main/notebook/demo.ipynb) notebook shows how to use TQA for your own time series model.


## Requirements
`numpy`, `torch`, `pandas`, `scipy`, `matplotlib`,  and `tqdm`
(`jupyter` and `notebook` if you want to use the notebook)
`env.yml` contains the full environment.

### Update 4/7/2023
I cached and analyzed the results of all my experiments. 
You'd need [persist-to-disk](https://pypi.org/project/persist-to-disk/) for this (Feel free to use it in your own research!).
If you don't need this feature, you can comment out the decorator (`@ptd.persistf`) in `main_experiments.py`.


### Bibtex
```
@inproceedings{NEURIPS2022_c8d2860e,
 author = {Lin, Zhen and Trivedi, Shubhendu and Sun, Jimeng},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {31017--31030},
 publisher = {Curran Associates, Inc.},
 title = {Conformal Prediction with Temporal Quantile Adjustments},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/c8d2860e1b51a1ffadc7ed0a06f8d8f5-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
