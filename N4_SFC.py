```python
# %% [markdown]
# """
# structure-function relationships
# """

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from netneurotools import datasets, stats, plotting, metrics
from scipy.stats import zscore, pearsonr
import seaborn as sns
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_random_state
from nilearn.datasets import fetch_atlas_schaefer_2018
from statsmodels.stats.multitest import multipletests
from netneurotools import datasets, freesurfer, stats
import os
import re


# %%
def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - float(SS_Residual) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (
        len(y) - X.shape[1] - 1
    )
    return adjusted_r_squared, SS_Residual, yhat


# %%
def cv_slr_distance_dependent(X, y, coords, train_pct=.75, metric='rsq'):
    '''
    Cross-validates linear regression using a distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1) = proportion of observations in training set
    metric = {'rsq', 'corr'}
    '''

    P = squareform(pdist(coords, metric="euclidean"))
    train_metric = []
    test_metric = []

    for i in range(len(y)):
        distances = P[i, :]
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])

        if metric == 'rsq':
            train_metric.append(
                get_reg_r_sq(X[train_idx, :], y[train_idx])[0]
            )

        elif metric == 'corr':
            rho, _ = pearsonr(
                mdl.predict(X[train_idx, :]), y[train_idx]
            )
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])

        if metric == 'rsq':
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum(
                (y[test_idx] - np.mean(y[test_idx])) ** 2
            )
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1 - (1 - r_squared) * (
                (len(y[test_idx]) - 1) /
                (len(y[test_idx]) - X.shape[1] - 1)
            )
            test_metric.append(adjusted_r_squared)

        elif metric == 'corr':
            rho, _ = pearsonr(yhat, y[test_idx])
            test_metric.append(rho)

    return train_metric, test_metric


# %%
def corr_spin(x, y, spins, nspins):
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    for i in range(nspins):
        null[i], _ = pearsonr(x[spins[:, i]], y)

    pval = (
        1 + sum(
            abs(null - np.mean(null)) >
            abs(rho - np.mean(null))
        )
    ) / (nspins + 1)

    return rho, pval


def get_perm_p(emp, null):
    return (
        1 + sum(
            abs(null - np.mean(null)) >
            abs(emp - np.mean(null))
        )
    ) / (len(null) + 1)


# %%
"""
Set-up
"""
path = '/Users/linqing/Desktop/Projects/2025_TAN/'  # replace your path

os.makedirs(
    os.path.join(path, 'results', 'N4_sfc'),
    exist_ok=True
)


# %%
nnodes = 400

annot = datasets.fetch_schaefer2018(
    version='fsaverage5',
    data_dir=None,
    url=None,
    resume=True,
    verbose=1
)['400Parcels7Networks']

coords, hemi = freesurfer.find_parcel_centroids(
    lhannot=annot.lh,
    rhannot=annot.rh,
    version='fsaverage5',
    surf='sphere',
    method='surface'
)

nspins = 1000

spins = stats.gen_spinsamples(
    coords,
    hemi,
    n_rotate=nspins,
    method='hungarian',
    seed=1234
)

eu = squareform(pdist(coords, metric='euclidean'))


# %% [markdown]
# Part 1: SFC analysis for group-level network


# %%
# Load data
sc_weighted = np.genfromtxt(
    path + 'data/Schaefer400_dwi.csv',
    delimiter=','
)
fc = np.genfromtxt(
    path + 'data/Schaefer400_rest.csv',
    delimiter=','
)
TAN = np.genfromtxt(
    path + 'data/Schaefer400_TAN.csv',
    delimiter=','
)

mask = np.triu(np.ones(nnodes), 1) > 0


# SFC analysis
co = metrics.communicability_wei(sc_weighted)

rsq_sc = np.zeros([nnodes, ])
rsq_r = np.zeros([nnodes, ])
train = np.zeros([nnodes, nnodes, 2])
test = np.zeros(train.shape)
rnull = np.zeros([nnodes, nspins])

for i in range(nnodes):
    print(i)

    y = fc[:, i]
    x1 = co[:, i]
    x2 = TAN[:, i]

    x_sc = zscore(x1).reshape(-1, 1)
    x_r = zscore(np.stack((x1, x2), axis=1))

    rsq_sc[i], _, yhat_sc = get_reg_r_sq(x_sc, y)
    rsq_r[i], _, yhat_rmn = get_reg_r_sq(x_r, y)

    train[i, :, 0], test[i, :, 0] = cv_slr_distance_dependent(
        x_sc, y, coords, metric='corr'
    )
    train[i, :, 1], test[i, :, 1] = cv_slr_distance_dependent(
        x_r, y, coords, metric='corr'
    )

    for s in range(nspins):
        x_rnull = zscore(
            np.stack((x1, x2[spins[:, s]]), axis=1)
        )
        rnull[i, s], _, _ = get_reg_r_sq(x_rnull, y)


rpvals = np.zeros([nnodes, ])

for i in range(nnodes):
    rpvals[i] = (
        1 + sum(rnull[i, :] > rsq_r[i])
    ) / (nspins + 1)

rpvals = multipletests(rpvals, method='fdr_bh')[1]


# Save results
pd.DataFrame(rnull).to_csv(
    path + 'results/N4_sfc/scfc_coupling_null.csv',
    index=False
)
pd.DataFrame(rpvals).to_csv(
    path + 'results/N4_sfc/scfc_coupling_rpval.csv',
    index=False
)
pd.DataFrame(rsq_r).to_csv(
    path + 'results/N4_sfc/scfc_coupling_r.csv',
    index=False
)
pd.DataFrame(rsq_sc).to_csv(
    path + 'results/N4_sfc/scfc_coupling_sc.csv',
    index=False
)

np.save(
    path + 'results/N4_sfc/scfc_coupling_train.npy',
    train
)
np.save(
    path + 'results/N4_sfc/scfc_coupling_test.npy',
    test
)


# Plot results
pvals = np.genfromtxt(
    path + 'results/N4_sfc/scfc_coupling_rpval.csv',
    delimiter=','
)
rsq_r = np.genfromtxt(
    path + 'results/N4_sfc/scfc_coupling_r.csv',
    delimiter=','
)
rsq_sc = np.genfromtxt(
    path + 'results/N4_sfc/scfc_coupling_sc.csv',
    delimiter=','
)

colors = np.where(rpvals < 0.05, '#F39C12', '#3498DB')

plt.ion()
fig, ax = plt.subplots()

plt.scatter(rsq_sc, rsq_r, c=colors, s=10)
plt.plot(rsq_sc, rsq_sc, 'k-', linewidth=0.5)

plt.xlabel('Adjusted $R^2$ from sc')
plt.ylabel('Adjusted $R^2$ from sc and TAN')

ax.set_aspect(
    1.0 / ax.get_data_ratio(),
    adjustable='box'
)

legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=r'$P_{\mathrm{spin}} < 0.05$',
        markerfacecolor='#F39C12',
        markersize=6
    )
]

plt.legend(
    handles=legend_elements,
    loc='lower right',
    frameon=False
)
plt.show()


# %% [markdown]
# Part 2: SFC analysis at individual level


# %%
# Get subject list for TAN
dwi_path = os.path.join(path, 'data', 'DWI_individual')

subject_list = sorted(list(set(
    re.match(r'(\d+)_sc\.csv', f).group(1)
    for f in os.listdir(dwi_path)
    if f.endswith('_sc.csv')
)))

rsq_sc_dict = {}
rsq_r_dict = {}
rpval_fdr_dict = {}

output_path = os.path.join(
    path,
    'results',
    'N4_sfc',
    'Individual_results'
)
os.makedirs(output_path, exist_ok=True)


for subject in subject_list:
    print(f'Processing {subject}...')

    try:
        sc_path = os.path.join(
            path, 'data', 'DWI_individual',
            f'{subject}_sc.csv'
        )
        fc_path = os.path.join(
            path, 'data', 'Rest_individual',
            f'{subject}_rest.csv'
        )
        rmn_path = os.path.join(
            path, 'data', 'TAN_individual',
            f'{subject}_TAN.csv'
        )

        sc_weighted = np.genfromtxt(
            sc_path, delimiter=','
        )
        fc = np.genfromtxt(
            fc_path, delimiter=','
        )
        TAN = np.genfromtxt(
            rmn_path, delimiter=','
        )

        co = metrics.communicability_wei(sc_weighted)

        rsq_sc = np.zeros([nnodes])
        rsq_r = np.zeros([nnodes])
        rnull = np.zeros([nnodes, nspins])

        for i in range(nnodes):
            y = fc[:, i]
            x1 = co[:, i]
            x2 = TAN[:, i]

            x_sc = zscore(x1).reshape(-1, 1)
            x_r = zscore(np.stack((x1, x2), axis=1))

            rsq_sc[i], _, yhat_sc = get_reg_r_sq(x_sc, y)
            rsq_r[i], _, yhat_rmn = get_reg_r_sq(x_r, y)

            for s in range(nspins):
                x2_perm = x2[spins[:, s]]
                x_rnull = zscore(
                    np.stack((x1, x2_perm), axis=1)
                )
                rnull[i, s], _, _ = get_reg_r_sq(
                    x_rnull, y
                )

        rpvals = np.zeros(nnodes)

        for i in range(nnodes):
            rpvals[i] = (
                1 + np.sum(rnull[i, :] > rsq_r[i])
            ) / (nspins + 1)

        rpvals_fdr = multipletests(
            rpvals,
            method='fdr_bh'
        )[1]

        rsq_sc_dict[subject] = rsq_sc
        rsq_r_dict[subject] = rsq_r
        rpval_fdr_dict[subject] = rpvals_fdr

        colors = np.where(
            rpvals_fdr < 0.05,
            '#F39C12',
            '#3498DB'
        )

        plt.ioff()
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

        ax.scatter(rsq_sc, rsq_r, c=colors, s=10)
        ax.plot(rsq_sc, rsq_sc, 'k-', linewidth=0.5)

        ax.set_xlabel('Adjusted $R^2$ from SC')
        ax.set_ylabel('Adjusted $R^2$ from SC + TAN')

        ax.set_aspect(
            1.0 / ax.get_data_ratio(),
            adjustable='box'
        )

        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=r'$P_{\mathrm{spin}} < 0.05$',
                markerfacecolor='#F39C12',
                markersize=6
            )
        ]

        ax.legend(
            handles=legend_elements,
            loc='lower right',
            frameon=False
        )

        plt.tight_layout()

        fig.savefig(
            os.path.join(output_path, f'{subject}_SFC.eps'),
            format='eps'
        )
        fig.savefig(
            os.path.join(output_path, f'{subject}_SFC.jpg'),
            format='jpg'
        )

        plt.close()

    except Exception as e:
        print(f'❌ Error processing {subject}: {e}')


rsq_sc_df = pd.DataFrame(rsq_sc_dict)
rsq_r_df = pd.DataFrame(rsq_r_dict)
rpval_fdr_df = pd.DataFrame(rpval_fdr_dict)

rsq_sc_df.insert(0, 'node', np.arange(nnodes))
rsq_r_df.insert(0, 'node', np.arange(nnodes))
rpval_fdr_df.insert(0, 'node', np.arange(nnodes))

rsq_sc_df.to_csv(
    os.path.join(output_path, 'rsq_sc_all.csv'),
    index=False
)
rsq_r_df.to_csv(
    os.path.join(output_path, 'rsq_r_all.csv'),
    index=False
)
rpval_fdr_df.to_csv(
    os.path.join(output_path, 'rpval_fdr_all.csv'),
    index=False
)

print('all finished！')


# %% [markdown]
# Part 3: SFC analysis for neurophysiology networks


# %%
networks = [
    'gene',
    'receptor',
    'laminar',
    'metabolic',
    'electrophysiological'
]

fc = np.genfromtxt(
    path + 'data/Schaefer400_rest.csv',
    delimiter=','
)
sc_weighted = np.genfromtxt(
    path + 'data/Schaefer400_dwi.csv',
    delimiter=','
)

co = metrics.communicability_wei(sc_weighted)

rsq_r_all = {}
rpvals_all = {}

rsq_sc = np.zeros(nnodes)

for net in networks:
    print(f'Processing network: {net}')

    Net = np.genfromtxt(
        path + f'3.SFC/Schaefer400_{net}.csv',
        delimiter=','
    )

    rsq_r = np.zeros(nnodes)
    rnull = np.zeros((nnodes, nspins))

    for i in range(nnodes):
        y = fc[:, i]
        x1 = co[:, i]
        x2 = Net[:, i]

        x_sc = zscore(x1).reshape(-1, 1)
        x_r = zscore(np.stack((x1, x2), axis=1))

        rsq_sc[i], _, _ = get_reg_r_sq(x_sc, y)
        rsq_r[i], _, _ = get_reg_r_sq(x_r, y)

        for s in range(nspins):
            x_rnull = zscore(
                np.stack((x1, x2[spins[:, s]]), axis=1)
            )
            rnull[i, s], _, _ = get_reg_r_sq(
                x_rnull, y
            )

    rpvals = np.array([
        (1 + np.sum(rnull[i, :] > rsq_r[i])) /
        (nspins + 1)
        for i in range(nnodes)
    ])

    rpvals_fdr = multipletests(
        rpvals,
        method='fdr_bh'
    )[1]

    rsq_r_all[net] = rsq_r
    rpvals_all[net] = rpvals_fdr

    # Plot
    colors = np.where(
        rpvals_fdr < 0.05,
        '#F39C12',
        '#3498DB'
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.scatter(rsq_sc, rsq_r, c=colors, s=10)
    plt.plot(rsq_sc, rsq_sc, 'k-', linewidth=0.5)

    plt.xlabel('Adjusted $R^2$ from SC')
    plt.ylabel(f'Adjusted $R^2$ from SC + {net.upper()}')
    ax.set_aspect(
        1.0 / ax.get_data_ratio(),
        adjustable='box'
    )

    plt.title(f'SFC with {net.upper()}')

    # Save plot
    fig.savefig(
        path + f'/results/N4_sfc/TAN_SFC_{net}.eps',
        format='eps'
    )
    fig.savefig(
        path + f'results/N4_sfc/TAN_SFC_{net}.jpg',
        format='jpg',
        dpi=300
    )

    plt.close()


# Save results
np.savetxt(
    path + 'results/N4_sfc/multi_SFC_rsq_r.csv',
    np.column_stack([
        rsq_r_all[net] for net in networks
    ]),
    delimiter=',',
    header=','.join(networks),
    comments=''
)

np.savetxt(
    path + 'results/N4_sfc/multi_SFC_rpvals.csv',
    np.column_stack([
        rpvals_all[net] for net in networks
    ]),
    delimiter=',',
    header=','.join(networks),
    comments=''
)
```
