import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt

from scipy.stats import median_absolute_deviation

#==============Simulation for the best case, where the even of "coverage" is temporally independent
def sim_ideal_coverage_one(T=20, p=0.9, N=1000, seed=0):
    np.random.seed(seed)
    cov_TS = pd.DataFrame(np.random.uniform(0, 1, (T, N)) < p)
    cov_TS = (cov_TS.cumsum().T / (cov_TS.index + 1)).T
    res = pd.DataFrame(np.sort(cov_TS, 1), index=cov_TS.index)
    return res

def get_raw_df(model, test_dataset, alpha, pred_in_seq=False, pred_kwargs={}, device='cuda:0', **kwargs):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1 if pred_in_seq else 32, shuffle=False)
    model = model.eval().to(device)
    res = {"yhat": [], 'lb': [], 'ub':[], 'y': []}
    extra = []
    with torch.no_grad():
        for _all_data in tqdm.tqdm(test_loader, desc='eval test'):
            _all_data = [_.to(device) for _ in _all_data]
            sequences, y, lengths_input, lengths_target = test_dataset._sep_data(_all_data)
            if pred_in_seq:
                raise NotImplementedError()
            else:
                _ = model.predict(sequences, y=y, alpha=alpha, **pred_kwargs)
                if len(_) == 2:
                    yhat, pi = _
                else:
                    yhat, pi = _[:2]
                    extra.append(_[2])
            res['yhat'].append(yhat)
            res['lb'].append(pi[:, 0])
            res['ub'].append(pi[:, 1])
            res['y'].append(y)

    res = {k: pd.DataFrame(torch.cat(v, 0).squeeze(-1).cpu().numpy()).stack() for k, v in res.items()}
    summ_df = pd.DataFrame(res).reset_index().rename(columns={"level_0": 'i', 'level_1': 't'})
    return summ_df


def _summ_cummean_cov_by_t(df):
    df = df.reindex()
    df['cov'] = (df['y'] <= df['ub']) & (df['y'] >= df['lb'])
    tdf1 = df.pivot_table(columns='i', index='t', values='cov').sort_index()
    return tdf1

def summ_perf(df, fill_inf='2max', min_t = 0, rescale_width=None, summ_by_T=False):
    if rescale_width is not None:
        df = df.reindex()
        df['ub'] = (df['ub'] - df['yhat']) * rescale_width + df['yhat']
        df['lb'] = (df['lb'] - df['yhat']) * rescale_width + df['yhat']
    summ_by_T_res = _summ_cummean_cov_by_t(df) if summ_by_T else None
    df = df[df['t'] >= min_t] #In some cases, we only look at after adapting a while
    df['cov'] = (df['y'] <= df['ub']) & (df['y'] >= df['lb'])
    df['w'] = (df['ub'] - df['lb'])
    if fill_inf is not None:
        if isinstance(fill_inf, str):
            if fill_inf == '2max':
                fill_inf = 2 * df['w'][df['w'].abs() < np.inf].max()
            else:
                raise NotImplementedError()
    else:
        fill_inf = np.inf

    w_res = {}
    w_res['median'] = df['w'].median()
    w_res['MAD'] = median_absolute_deviation(df['w'])
    w_res['mean'] = df['w'].replace([np.inf], fill_inf, inplace=False).mean()
    w_res['std'] = df['w'].replace([np.inf], fill_inf, inplace=False).std()

    ts_mean = df.groupby('i')['cov'].mean().sort_values()
    cov_res = {}
    cov_res['mean'] = ts_mean.mean() #ts_mean_stats['mean']
    cov_res['std'] = ts_mean.std()# ts_mean_stats['std']
    cov_res['10%'] = ts_mean.iloc[int(np.round(0.1 * len(ts_mean)))] #ts_mean_stats['25%']
    cov_res['5%'] = ts_mean.iloc[int(np.round(0.05 * len(ts_mean)))]  # ts_mean_stats['25%']
    cov_res['min'] = ts_mean.iloc[0] #ts_mean_stats['min']
    cov_res['mean  10% of mean ts_cov'] = ts_mean.iloc[:int(np.round(0.1 * len(ts_mean)))].mean()
    return w_res, ts_mean, summ_by_T_res

def plot_coverage(cov_res, nx=20, ax=None, perc=10, rename_cols={}, normalize_x = 0.1, title=None, linestyle={}, legend_loc='best'):
    colors = ['red', 'green', 'blue', 'orange', 'black', 'purple', 'gray', 'yellow', 'maroon', 'olive', 'pink', 'aqua']
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1)
    x = np.arange(nx)
    if normalize_x is not None: x = x / (nx - 1) * normalize_x

    for i, (method, cov) in enumerate(cov_res.items()):
        ax.plot(x, cov.mean(0).iloc[:nx], color=colors[i], alpha=1, label=rename_cols.get(method, method), linestyle=linestyle.get(method, None))
        if perc is not None:
            lo = np.percentile(cov, perc, axis=0)[:nx]
            hi = np.percentile(cov, 100 - perc, axis=0)[:nx]
            ax.fill_between(x, lo, hi, color=colors[i], alpha=.1, label=None)
    if title:
        ax.set_title(title)
    ax.legend(loc=legend_loc)


def compare_tail_coverage(covs, tail_n = 10, base_method='CFRNN', quiet=False):
    res = {}
    for method, cov in covs.items():
        res[method] = cov.iloc[:, :tail_n].T.mean()
    for method, cov_mean in res.items():
        from scipy import stats
        if quiet: continue
        if method != base_method:
            ttest = stats.ttest_rel(cov_mean, res[base_method])
            ttest_ind = stats.ttest_ind(cov_mean, res[base_method])
            print(f"{method}: ind pval:{ttest_ind.pvalue}, paired pval: {ttest.pvalue}")
    res = pd.DataFrame(res)
    return res

def compare_efficiency(widths, covs, base_method='CFRNN'):
    res = {}
    for method, w in widths.items():
        res[method] = w['mean'].values / covs[method].mean(1).values
    res = pd.DataFrame(res)
    return res
