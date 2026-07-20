
import pandas as pd
from netneurotools import datasets, freesurfer, stats

nnodes=400
annot= datasets.fetch_schaefer2018(version='fsaverage5', data_dir=None, url=None, resume=True, verbose=1)['400Parcels7Networks']
coords, hemi = freesurfer.find_parcel_centroids(lhannot=annot.lh,
                                                rhannot=annot.rh,
                                                version='fsaverage5',
                                                surf='sphere',
                                                method='surface')
nspins = 1000
spins = stats.gen_spinsamples(coords, hemi, method='hungarian', check_duplicates=True, exact=False, seed=665, verbose=False, return_cost=False)
spins = spins + 1
pd.DataFrame(spins).to_csv(path + 'data/1000Null_Schaefer400.csv', index=False, header=False)
