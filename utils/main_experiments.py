import os.path

import pandas as pd
import persist_to_disk as ptd
import torch

import data.online_dataset as dld
import models
import models.base_rnn as base_rnn
import models.dprnn as dprnn
import models.qrnn as qrnn
import utils.evaluate as eval_utils
import utils.utils1 as utils
from _settings import (BASE_MODEL_PATH, COVID_NAME, EEG_NAME, MIMIC_NAME,
                       GEFCom_NAME)

DEFAULT_PARAMETERS = {
    "batch_size": 128,
    "embedding_size": 32,
    "coverage": 0.9,
    "lr": 1e-3,
    "rnn_mode": "LSTM",
}


RNNS = {"RNN": base_rnn.MyRNN,
        "QRNN": qrnn.QRNN,
        'DPRNN': dprnn.DPRNN,
        }



BASELINES = {"CFRNN": models.CFRNN,
             'TQA-E': models.TQA_E,
             'TQA-B': models.TQA_B,

             # Other variants of TQA-B used in the Appendix
             'TQA-B-A-S': models.TQA_B_AS,
             'TQA-B-A-R': models.TQA_B_AR,
             'TQA-B-C-R': models.TQA_B_CR,

             'CQRNN': models.CQRNN,
             'LASplit': models.LASplit,
        }

DEFAULT_BASELINE_PRED_KWARGS = {
                                'TQA-E': {'two_sided': False},
                                }

EPOCHS = {MIMIC_NAME: 200, COVID_NAME: 1000, GEFCom_NAME:1000, EEG_NAME: 100}

def is_conformal(baseline):
    return baseline in BASELINES.keys()

def is_conformally_trained(baseline):
    return is_conformal(baseline) or '-C' in baseline

def train_RNN(dataset, baseline, params=None, seed=0, fit_kwargs={}, gpu_id=0):

    params = DEFAULT_PARAMETERS.copy() if params is None else params
    params["epochs"] = EPOCHS[dataset.split("-")[0]]

    horizon = dld.get_horizon(dataset)
    utils.set_all_seeds(seed)

    device = utils.gpuid_to_device(gpu_id)
    if baseline.startswith("RNN"):
        baseline, rnn_mode = baseline.split("-")
        rnn_path = os.path.join(BASE_MODEL_PATH, f'{dataset}-{seed}-{rnn_mode}-{params["embedding_size"]}_{params["epochs"]}_{params["lr"]}.pt')
        if os.path.isfile(rnn_path): return rnn_path

        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
        model = RNNS[baseline](embedding_size=params["embedding_size"], input_size = train_dataset[0][0].shape[1], output_size=train_dataset[0][1].shape[1], horizon=horizon, rnn_mode=rnn_mode, path=rnn_path).to(device)
        model.fit(train_dataset, batch_size=params['batch_size'], epochs=params["epochs"], lr=params["lr"], val_dataset = calibration_dataset, device=device, tags=[baseline], **fit_kwargs)
    elif baseline.startswith('MADRNN'): #Requires a RNN first
        base_rnn_path = train_RNN(dataset, baseline.replace("MADRNN", "RNN"), params, seed, fit_kwargs, gpu_id)
        baseline, rnn_mode = baseline.split("-")
        rnn_path = os.path.join(BASE_MODEL_PATH, f'{dataset}-{seed}-MAD{rnn_mode}-{params["embedding_size"]}_{params["epochs"]}_{params["lr"]}.pt')
        if os.path.isfile(rnn_path): return rnn_path

        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)

        model = base_rnn.ResidRNN(base_rnn_path, rnn_path).to(device)
        model.fit(train_dataset, batch_size=params['batch_size'], epochs=params["epochs"], lr=params["lr"], val_dataset = calibration_dataset, device=device, tags=[baseline], **fit_kwargs)
    elif baseline.startswith('DPRNN'):
        params["lr"] *= 3 #Dropout trains a bit slower

        suffix = f'-{params["embedding_size"]}_{params["epochs"]}_{params["lr"]}_conf'
        rnn_path = os.path.join(BASE_MODEL_PATH, f'{dataset}-{seed}-{baseline}{suffix}.pt')
        if os.path.isfile(rnn_path): return rnn_path

        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
        model = RNNS[baseline](input_size = train_dataset[0][0].shape[1], output_size=1, **params).to(device)
        print(params)
        model.fit(train_dataset, batch_size=params['batch_size'], val_dataset = calibration_dataset, device=device, **fit_kwargs)
        model.eval()
        torch.save(model, rnn_path)
    else: #Quantile RNN
        suffix = f'-{params["embedding_size"]}_{params["epochs"]}_{params["lr"]}'
        if baseline == 'QRNN': suffix = f"{suffix}_{params['coverage']}_conf"
        rnn_path = os.path.join(BASE_MODEL_PATH, f'{dataset}-{seed}-{baseline}{suffix}.pt')
        if os.path.isfile(rnn_path): return rnn_path

        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
        assert params['rnn_mode'] == 'LSTM', "I did not include this in the model save path.."
        model = RNNS[baseline](input_size = train_dataset[0][0].shape[1], output_size=1, **params).to(device)
        model.fit(train_dataset, batch_size=params['batch_size'], val_dataset = calibration_dataset, device=device, **fit_kwargs)
        model.eval()
        torch.save(model, rnn_path)

    return rnn_path



def get_calibrated_model(dataset, baseline, params=None, seed=0, device='cpu'):
    conformal = is_conformal(baseline)

    horizon = dld.get_horizon(dataset)
    utils.set_all_seeds(seed)

    if baseline == 'LASplit':
        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
        rnn_path = train_RNN(dataset, baseline='MADRNN-LSTM', params=params, seed=seed)
        model = BASELINES[baseline](base_model_path=rnn_path, device=device)
        model = model.to(device)
        model.calibrate(calibration_dataset, device=device)
        return model.eval()
    elif conformal:
        train_dataset, calibration_dataset, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
        base_rnn_name = 'QRNN' if baseline == 'CQRNN' else 'RNN-LSTM'
        rnn_path = train_RNN(dataset, baseline=base_rnn_name, params=params,  seed=seed)
        model = BASELINES[baseline](base_model_path=rnn_path)
        model = model.to(device)
        model.calibrate(calibration_dataset, device=device)
    else:
        rnn_path = train_RNN(dataset, baseline=baseline, params=params, seed=seed)
        model = torch.load(rnn_path)
        model = model.to(device)
    model = model.eval()
    return model

@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['gpu_id'], groupby=['dataset', 'baseline'])
def get_results_general(dataset, baseline, seed=0, params=None, correct_alpha=False, alpha=None,
                        pred_kwargs={},
                        **kwargs):
    kwargs = kwargs.copy()
    device = utils.gpuid_to_device(kwargs.pop('gpu_id', -1))
    model = get_calibrated_model(dataset, baseline, params=params, seed=seed, device=device)

    if alpha is None: alpha = 1 - params['coverage']
    horizon = dld.get_horizon(dataset)
    if correct_alpha: alpha /= horizon
    _, _, test_dataset = dld.get_default_data(dataset, horizon=horizon, seed=seed)
    default_pred_kwargs = DEFAULT_BASELINE_PRED_KWARGS.get(baseline, {})
    pred_kwargs = utils.merge_dict_inline(default_pred_kwargs, pred_kwargs) #the ordering matters

    df = eval_utils.get_raw_df(model, test_dataset, alpha,pred_kwargs=pred_kwargs, device=device)
    df['i'] = df['i'].map(lambda i: f"{seed}-{i}")
    return df

def summ_results(baseline, dataset, seeds=list(range(5)), correct_alpha=False, alpha=0.1,
                                  pred_kwargs={}, rescale_width=None,  **kwargs):
    w_df = []
    cov_df = []
    cummean_cov_df = []
    kwargs = kwargs.copy()
    min_t = kwargs.pop('min_t', 0)
    summ_by_T = kwargs.pop('summ_by_T', False)
    for seed in seeds:
        df = get_results_general(dataset, baseline, seed, correct_alpha=correct_alpha, alpha=alpha, pred_kwargs=pred_kwargs, #cache=3,
                                 **kwargs)
        w_res, ts_mean, cummean_cov = eval_utils.summ_perf(df, min_t=min_t, rescale_width=rescale_width, summ_by_T=summ_by_T)
        cov_df.append(ts_mean.reset_index()['cov'])
        w_df.append(pd.Series(w_res))
        cummean_cov_df.append(cummean_cov)
    w_df = pd.DataFrame(w_df)
    cov_df = pd.DataFrame(cov_df)
    return w_df, cov_df, cummean_cov_df

if __name__ == '__main__':
    import utils.main_experiments as tbr
    import utils.utils1 as utils
    NSEEDS = 2
    o_base = utils.TaskPartitioner(seed=5)
    o2_base = utils.TaskPartitioner(seed=6)
    o = utils.TaskPartitioner(seed=7)

    datasets = ['GEFCom-R'] #+ [ 'COVID', 'MIMIC', 'claim', 'GEFCom', 'EEG']
    for seed in range(NSEEDS):
        fit_kwargs = {"log": seed < 2}
        for dataset in datasets:
            o_base.add_task(tbr.train_RNN, dataset, 'QRNN', seed=seed, fit_kwargs=fit_kwargs)
            o_base.add_task(tbr.train_RNN, dataset, 'RNN-LSTM', seed=seed, fit_kwargs=fit_kwargs)
            o2_base.add_task(tbr.train_RNN, dataset, 'MADRNN-LSTM', seed=seed, fit_kwargs=fit_kwargs)
            o_base.add_task(tbr.train_RNN, dataset, 'DPRNN', seed=seed, fit_kwargs=fit_kwargs)
    o_base.run_multi_process(8, process_kwarg='gpu_id', cache_only=True)
    o2_base.run_multi_process(8, process_kwarg='gpu_id', cache_only=True)

