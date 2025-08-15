import numpy as np
import matplotlib.pyplot as plt

def plot_dt_hist(timestamps_us, hz, jump_factor=1.5):
    dt = np.diff(timestamps_us.astype(np.int64))
    if len(dt) == 0:
        raise ValueError("Not enough samples for dt histogram")
    exp = int(round(1e6 / hz))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(dt, bins=100)
    ax.axvline(exp, linestyle="--")
    ax.axvline(jump_factor*exp, linestyle="--")
    ax.set_title(f"dt histogram @ {hz}Hz (expected≈{exp}us)")
    ax.set_xlabel("microseconds"); ax.set_ylabel("count")
    fig.tight_layout(); return fig

def plot_channel_hist(df, title=None):
    fig, axs = plt.subplots(2, 3, figsize=(12,6))
    cols = ["gx","gy","gz","ax","ay","az"]
    for i,c in enumerate(cols):
        r, k = divmod(i,3)
        axs[r,k].hist(df[c].values, bins=100)
        axs[r,k].set_title(c)
    if title: fig.suptitle(title)
    fig.tight_layout(); return fig

def plot_segment_preview(df, n_points=1000, title=None):
    T = min(len(df), n_points)
    t = np.arange(T)
    fig, axs = plt.subplots(2,1, figsize=(12,6), sharex=True)
    axs[0].plot(t, df["gx"].values[:T], label="gx")
    axs[0].plot(t, df["gy"].values[:T], label="gy")
    axs[0].plot(t, df["gz"].values[:T], label="gz")
    axs[0].legend(); axs[0].set_ylabel("gyro")
    axs[1].plot(t, df["ax"].values[:T], label="ax")
    axs[1].plot(t, df["ay"].values[:T], label="ay")
    axs[1].plot(t, df["az"].values[:T], label="az")
    axs[1].legend(); axs[1].set_ylabel("acc")
    axs[1].set_xlabel("sample")
    if title: axs[0].set_title(title)
    fig.tight_layout(); return fig