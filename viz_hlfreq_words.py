import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

df = pandas.DataFrame.from_csv('hl_freq_tests.csv', index_col=None)
max_hcut = max(df['high_idx'])
max_lcut = max(df['low_idx'])
#max_hcut = max(df['highest_freq'])
#max_lcut = max(df['lowest_freq'])


hl_scores = np.zeros((max_hcut+1, max_lcut+1))
hl_ll = np.zeros((max_hcut+1, max_lcut+1))
hl_nme = np.zeros((max_hcut+1, max_lcut+1))

for row_idx in range(len(df['score'])):
    h_idx = df['high_idx'][row_idx]
    l_idx = df['low_idx'][row_idx]
    #h_idx = df['highest_freq'][row_idx]
    #l_idx = df['lowest_freq'][row_idx]
    score = df['score'][row_idx]
    ll = df['loglikelihood'][row_idx]
    nme = df['mean_n_tied_best'][row_idx]
    hl_scores[h_idx][l_idx] = score
    hl_ll[h_idx][l_idx] = ll
    hl_nme[h_idx][l_idx] = nme


### LOG LIKELIHOOD
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlim((1,max_hcut+1))
ax.set_yscale('log')
ax.set_ylim((1,max_lcut+1))
heatmap = ax.pcolormesh(hl_ll, cmap=plt.cm.Oranges_r)
#ax.set_yticks(np.arange(alphameans_matrix.shape[1])+0.5, minor=False)
#ax.set_yticklabels(all_alpha, minor=False)
plt.show()


### SCORES
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlim((1,max_hcut+1))
ax.set_yscale('log')
ax.set_ylim((1,max_lcut+1))
#plot(df['ncols'], df['score'],'o')

scores_gaus = ndimage.filters.gaussian_filter(hl_scores, 2, mode='nearest')

heatmap = ax.pcolormesh(hl_scores, cmap=plt.cm.Greys)
#ax.set_yticks(np.arange(alphameans_matrix.shape[1])+0.5, minor=False)
#ax.set_yticklabels(all_alpha, minor=False)
plt.show()
