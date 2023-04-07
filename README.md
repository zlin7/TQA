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
All my experiments were run by caching down the results and then reading/analyzing them.
You need [persist-to-disk](https://pypi.org/project/persist-to-disk/) for this (Feel free to use it in your own research!).
If you don't need this functionality, you can comment out the decorator (`@ptd.persistf`) in `main_experiments.py`.
