from scipy.stats import wasserstein_distance, energy_distance
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def calc_wasserstein_score(source_vals, proc_vals, inverse_sign=False, custom_scaler=None):
    if custom_scaler is None:
        scaler = MinMaxScaler()
        source_vals_scaled = scaler.fit_transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
    else:
        source_vals_scaled = custom_scaler.transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = custom_scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
    k = 1.
    if inverse_sign:
        k = -1.
    return k * wasserstein_distance(source_vals_scaled.squeeze(), proc_vals_scaled.squeeze()) * np.sign(np.mean(proc_vals_scaled) - np.mean(source_vals_scaled))


def energy_distance_score(source_vals, proc_vals, inverse_sign=False, custom_scaler=None):
    if custom_scaler is None:
        scaler = MinMaxScaler()
        source_vals_scaled = scaler.fit_transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
    else:
        source_vals_scaled = custom_scaler.transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = custom_scaler.transform(np.array(proc_vals.values).reshape(-1, 1))

    k = 1.
    if inverse_sign:
        k = -1.
    return k * energy_distance(source_vals_scaled.squeeze(), proc_vals_scaled.squeeze()) * np.sign(np.mean(proc_vals_scaled) - np.mean(source_vals_scaled))


def normalized_absolute_gain(source_vals, proc_vals, inverse_sign=False, custom_scaler=None):
    if custom_scaler is None:
        scaler = MinMaxScaler()
        source_vals_scaled = scaler.fit_transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
    else:
        source_vals_scaled = custom_scaler.transform(np.array(source_vals.values).reshape(-1, 1))
        proc_vals_scaled = custom_scaler.transform(np.array(proc_vals.values).reshape(-1, 1))

    k = 1.
    if inverse_sign:
        k = -1.
    return k * (proc_vals_scaled.squeeze() - source_vals_scaled.squeeze())


def normalized_relative_gain(source_vals, proc_vals, inverse_sign=False, custom_scaler=None):
    if custom_scaler is None:
        scaler = MinMaxScaler()
        source_vals_scaled = scaler.fit_transform(np.array(source_vals.values).reshape(-1, 1)) + 1.0
        proc_vals_scaled = scaler.transform(np.array(proc_vals.values).reshape(-1, 1)) + 1.0
    else:
        source_vals_scaled = custom_scaler.transform(np.array(source_vals.values).reshape(-1, 1)) + 1.0
        proc_vals_scaled = custom_scaler.transform(np.array(proc_vals.values).reshape(-1, 1)) + 1.0

    k = 1.
    if inverse_sign:
        k = -1.
    return (k * (proc_vals_scaled - source_vals_scaled) / (source_vals_scaled)).reshape(-1)


def robustness_score(source_vals, proc_vals, inverse_sign=False, beta1=1., beta2=0., eps=1e-6, normalize=True, custom_scaler=None):
    if normalize:
        if custom_scaler is None:
            scaler = MinMaxScaler()
            source_vals_scaled = scaler.fit_transform(np.array(source_vals.values).reshape(-1, 1))
            proc_vals_scaled = scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
        else:
            source_vals_scaled = custom_scaler.transform(np.array(source_vals.values).reshape(-1, 1))
            proc_vals_scaled = custom_scaler.transform(np.array(proc_vals.values).reshape(-1, 1))
    else:
        source_vals_scaled = source_vals
        proc_vals_scaled = proc_vals
    denom = np.abs(source_vals_scaled - proc_vals_scaled) + eps
    numer = np.maximum(beta1 - source_vals_scaled, source_vals_scaled - beta2)
    scores = np.log10(numer / denom)
    return scores.reshape(-1)


def relative_gain_classic(source_vals, proc_vals, inverse_sign=False, eps=1e-6, custom_scaler=None):
    return (proc_vals - source_vals) / (source_vals + eps)


method_name_to_func = {
    'robustness_score': robustness_score,
    'relative_gain_classic': relative_gain_classic,
    'normalized_relative_gain': normalized_relative_gain,
    'normalized_absolute_gain': normalized_absolute_gain,
    'wasserstein_score': calc_wasserstein_score,
    'energy_distance_score': energy_distance_score,
}
