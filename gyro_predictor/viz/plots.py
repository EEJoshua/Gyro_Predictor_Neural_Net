import numpy as np, matplotlib.pyplot as plt
def plot_timeseries(true, pred, title=None, n_points=1000):
    T = min(len(true), n_points); t = np.arange(T)
    fig, axs = plt.subplots(4,1, figsize=(10,8), sharex=True)
    for i, ax in enumerate(axs[:3]):
        ax.plot(t, true[:T, i], label='true')
        ax.plot(t, pred[:T, i], label='pred', linestyle='--')
        ax.set_ylabel(['gx','gy','gz'][i]); ax.legend()
    err = np.linalg.norm(pred[:T] - true[:T], axis=1)
    axs[3].plot(t, err, label='L2 error'); axs[3].set_ylabel('L2')
    if title: axs[0].set_title(title)
    axs[-1].set_xlabel('sample'); plt.tight_layout(); return fig
def plot_error_hist(err_l2, title=None, bins=50):
    fig, ax = plt.subplots(figsize=(8,4)); ax.hist(err_l2, bins=bins)
    ax.set_xlabel('L2 error'); ax.set_ylabel('count'); 
    if title: ax.set_title(title); plt.tight_layout(); return fig
def plot_bar(metric_dict, title=None):
    import numpy as np
    ks = sorted(set(k for (_,k) in metric_dict.keys()))
    archs = sorted(set(a for (a,_) in metric_dict.keys()))
    fig, ax = plt.subplots(figsize=(10,5)); width=0.2
    for i,a in enumerate(archs):
        vals = [metric_dict.get((a,k), np.nan) for k in ks]
        ax.bar([k + i*width for k in ks], vals, width=width, label=a)
    ax.set_xticks([k + (len(archs)-1)*width/2 for k in ks]); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('k steps ahead'); ax.set_ylabel('metric')
    if title: ax.set_title(title); ax.legend(); plt.tight_layout(); return fig

def plot_deltas(true, pred, title=None, n_points=1000):
    import numpy as np, matplotlib.pyplot as plt
    T = min(len(true), n_points)
    diff = pred[:T] - true[:T]
    t = np.arange(T)
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, diff[:, i])
        ax.set_ylabel(['Δgx','Δgy','Δgz'][i])
    if title: axs[0].set_title(title)
    axs[-1].set_xlabel('sample')
    fig.tight_layout(); return fig