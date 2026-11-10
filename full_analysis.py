#!/usr/bin/env python3
"""
Time-Series Forecasting Project
Dataset: Daily Minimum Temperatures in Melbourne (1981-1990)
Decision context: nightly heating planning / frost-risk management

All models are evaluated with ONE-STEP-AHEAD forecasts on the same
chronological 70/15/15 split. Random seeds are fixed so the numbers in
results/ can be reproduced.
"""
import os, json, itertools, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
tf.keras.utils.set_random_seed(SEED); tf.config.experimental.enable_op_determinism()
DATA = os.environ.get('DATA_CSV', 'daily_min_temp_clean.csv')
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ---------- chart style: black text, serif (Cambria if installed) ----------
avail = {f.name for f in font_manager.fontManager.ttflist}
serif = next((f for f in ['Cambria', 'Caladea', 'DejaVu Serif'] if f in avail), 'serif')
plt.rcParams.update({
    'font.family': serif, 'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'text.color': 'black', 'axes.labelcolor': 'black',
    'axes.edgecolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
C_ACT, C_PRED, C_BAND, C_ALT = '#1a1a1a', '#c0392b', '#2e75b6', '#e08a00'
OUT = {}

# ---------------- data ----------------
df = pd.read_csv(DATA, parse_dates=['Date'], index_col='Date').asfreq('D')
series = df['Temp']
OUT['n'] = int(len(series)); OUT['mean'] = float(series.mean()); OUT['std'] = float(series.std())
OUT['min'] = float(series.min()); OUT['max'] = float(series.max())
OUT['days_below_8'] = int((series < 8).sum())
mon = series.groupby(series.index.month).agg(['mean', 'std', 'min', 'max'])
OUT['monthly'] = mon.round(2).to_dict()
OUT['yearly_mean'] = series.groupby(series.index.year).mean().round(2).to_dict()

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(series.index, series.values, color='#7f7f7f', lw=0.6, label='Daily minimum')
ax.plot(series.rolling(30, center=True).mean(), color=C_PRED, lw=1.8, label='30-day rolling mean')
ax.axhline(8, color=C_BAND, ls='--', lw=1.2, label='8 °C heating threshold')
ax.set_ylabel('Temperature (°C)'); ax.set_title('Daily minimum temperature, Melbourne, 1981–1990')
ax.legend(ncol=3, loc='upper center'); ax.set_ylim(-1, 30)
plt.savefig('plots/00_series.png'); plt.close()

# ---------------- decomposition ----------------
dec = seasonal_decompose(series, model='additive', period=365)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for a, (s, t) in zip(axes, [(dec.observed, 'Observed'), (dec.trend, 'Trend (365-day centred moving average)'),
                            (dec.seasonal, 'Seasonal component (period = 365 days)'), (dec.resid, 'Residual')]):
    a.plot(s.index, s.values, color=C_ACT, lw=0.8); a.set_title(t, loc='left'); a.set_ylabel('°C')
plt.tight_layout(); plt.savefig('plots/01_decomposition.png'); plt.close()
resid = dec.resid.dropna()
OUT['decomp'] = {
    'trend_min': float(dec.trend.min()), 'trend_max': float(dec.trend.max()),
    'seas_min': float(dec.seasonal.min()), 'seas_max': float(dec.seasonal.max()),
    'resid_std': float(resid.std()),
    'var_share_seasonal': float(dec.seasonal.var() / series.var()),
    'var_share_resid': float(resid.var() / series.var()),
    'resid_acf1': float(acf(resid, nlags=10)[1]), 'resid_acf2': float(acf(resid, nlags=10)[2]),
    'resid_acf7': float(acf(resid, nlags=10)[7]),
    'resid_ljungbox_p_lag10': float(acorr_ljungbox(resid, lags=[10])['lb_pvalue'].iloc[0]),
}

# ---------------- ACF / PACF ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
plot_acf(series, lags=60, ax=axes[0], title='ACF (lags 0–60)', color=C_ACT, vlines_kwargs={'colors': C_ACT})
plot_pacf(series, lags=60, ax=axes[1], title='PACF (lags 0–60)', method='ywm', color=C_ACT, vlines_kwargs={'colors': C_ACT})
for a in axes: a.set_xlabel('Lag (days)')
plt.tight_layout(); plt.savefig('plots/02_acf_pacf.png'); plt.close()
a_long = acf(series, nlags=800)
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(range(801), a_long, color=C_ACT, lw=1.2); ax.axhline(0, color='black', lw=0.6)
for x in (182, 365, 547, 730): ax.axvline(x, color=C_BAND, ls=':', lw=1)
ax.set_xlabel('Lag (days)'); ax.set_ylabel('Autocorrelation'); ax.set_title('ACF over 800 lags: the annual cycle')
plt.savefig('plots/02b_acf_long.png'); plt.close()
from statsmodels.tsa.stattools import pacf
p_ = pacf(series, nlags=30, method='ywm')
OUT['acf'] = {str(k): float(a_long[k]) for k in (1, 2, 3, 7, 14, 30, 60, 182, 365, 730)}
OUT['pacf'] = {str(k): float(p_[k]) for k in range(1, 16)}
OUT['pacf_ci'] = float(1.96 / np.sqrt(len(series)))

# ---------------- stationarity ----------------
def adf(s):
    r = adfuller(s.dropna()); return {'stat': float(r[0]), 'p': float(r[1]), 'lags': int(r[2]), 'crit1': float(r[4]['1%'])}
OUT['adf_original'] = adf(series); OUT['adf_diff'] = adf(series.diff())
k = kpss(series, regression='c', nlags='auto'); OUT['kpss_original'] = {'stat': float(k[0]), 'p': float(k[1])}
k = kpss(series.diff().dropna(), regression='c', nlags='auto'); OUT['kpss_diff'] = {'stat': float(k[0]), 'p': float(k[1])}

# ---------------- split ----------------
n = len(series); tr_end = int(n * 0.70); va_end = int(n * 0.85)
train, val, test = series.iloc[:tr_end], series.iloc[tr_end:va_end], series.iloc[va_end:]
OUT['split'] = {'train': [len(train), str(train.index[0].date()), str(train.index[-1].date())],
                'val': [len(val), str(val.index[0].date()), str(val.index[-1].date())],
                'test': [len(test), str(test.index[0].date()), str(test.index[-1].date())]}
OUT['test_stats'] = {'mean': float(test.mean()), 'std': float(test.std()), 'min': float(test.min()),
                     'days_below_8': int((test < 8).sum()), 'days_below_3': int((test < 3).sum()),
                     'zeros': int((test == 0).sum())}
OUT['train_mean'] = float(train.mean())
fig, ax = plt.subplots(figsize=(12, 3.6))
for s, c, l in [(train, '#4d4d4d', f'Training ({len(train)} days)'), (val, C_ALT, f'Validation ({len(val)} days)'),
                (test, C_BAND, f'Test ({len(test)} days)')]:
    ax.plot(s.index, s.values, color=c, lw=0.7, label=l)
ax.set_ylabel('Temperature (°C)'); ax.set_title('Chronological train / validation / test split'); ax.legend(ncol=3, loc='upper center')
ax.set_ylim(-1, 31); plt.savefig('plots/02c_split.png'); plt.close()

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'MAE': float(mean_absolute_error(y, p)),
            'MAPE': float(np.mean(np.abs((y - p) / y)) * 100), 'R2': float(r2_score(y, p))}

def plot_forecast(pred, title, fn, zoom=('1990-06-01', '1990-08-31')):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1.7, 1]})
    axes[0].plot(test.index, test.values, color=C_ACT, lw=1.0, label='Actual')
    axes[0].plot(test.index, pred, color=C_PRED, lw=1.0, alpha=0.9, label='One-step-ahead forecast')
    axes[0].set_title(title, loc='left'); axes[0].set_ylabel('Temperature (°C)'); axes[0].legend(ncol=2, loc='upper center')
    axes[0].set_ylim(-1, 26)
    z = test.loc[zoom[0]:zoom[1]]; pz = pd.Series(pred, index=test.index).loc[zoom[0]:zoom[1]]
    axes[1].plot(z.index, z.values, color=C_ACT, lw=1.4, marker='o', ms=2.5)
    axes[1].plot(pz.index, pz.values, color=C_PRED, lw=1.4, marker='o', ms=2.5)
    axes[1].axhline(8, color=C_BAND, ls='--', lw=1); axes[1].set_title('Zoom: winter 1990 (Jun–Aug)', loc='left')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.tight_layout(); plt.savefig(f'plots/{fn}'); plt.close()

results, preds = {}, {}

# ---------------- baselines ----------------
preds['Climatology'] = np.full(len(test), train.mean())
preds['Persistence'] = series.shift(1).loc[test.index].values
for k_ in ('Climatology', 'Persistence'): results[k_] = metrics(test, preds[k_])

# ---------------- classical models: fit on train, one-step-ahead on test ----------------
def one_step(res, exog_full=None):
    """Keep the parameters estimated on the training set, run the filter over the
    whole series and read off the one-step-ahead predictions for the test window."""
    full = res.apply(series, exog=exog_full) if exog_full is not None else res.apply(series)
    return full.predict(start=test.index[0], end=test.index[-1], dynamic=False).values

multi = {}
print('MA(7)...'); ma = ARIMA(train, order=(0, 0, 7)).fit()
preds['MA(7)'] = one_step(ma); multi['MA(7)'] = metrics(test, ma.forecast(len(val) + len(test)).values[-len(test):])
print('AR(14)...'); ar = ARIMA(train, order=(14, 0, 0)).fit()
preds['AR(14)'] = one_step(ar); multi['AR(14)'] = metrics(test, ar.forecast(len(val) + len(test)).values[-len(test):])
OUT['ar_params'] = {k: float(v) for k, v in ar.params.items()}
OUT['ma_params'] = {k: float(v) for k, v in ma.params.items()}
OUT['aic'] = {'MA(7)': float(ma.aic), 'AR(14)': float(ar.aic)}

print('ARIMA order search (AIC on training set)...')
grid = []
for p, d, q in itertools.product(range(4), (0, 1), range(4)):
    try:
        r = ARIMA(train, order=(p, d, q)).fit(); grid.append({'p': p, 'd': d, 'q': q, 'AIC': float(r.aic)})
    except Exception as e:
        pass
grid = sorted(grid, key=lambda g: g['AIC']); OUT['arima_grid'] = grid
b = grid[0]; best_order = (b['p'], b['d'], b['q']); arima_name = f'ARIMA({b["p"]},{b["d"]},{b["q"]})'
arima = ARIMA(train, order=best_order).fit(); preds[arima_name] = one_step(arima)
multi[arima_name] = metrics(test, arima.forecast(len(val) + len(test)).values[-len(test):])
OUT['arima_name'] = arima_name; OUT['aic'][arima_name] = float(arima.aic)
lb = acorr_ljungbox(arima.resid[5:], lags=[10, 20]); OUT['arima_resid_lb_p'] = [float(x) for x in lb['lb_pvalue']]

print('SARIMA weekly...')
sar_name = 'SARIMA(1,1,1)(1,0,1,7)'
sar = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
preds[sar_name] = one_step(sar); OUT['aic'][sar_name] = float(sar.aic)
OUT['sarima_params'] = {k: [float(v), float(sar.pvalues[k])] for k, v in sar.params.items()}

print('ARIMA + annual Fourier terms...')
doy = np.arange(n)
def fourier(K):
    return pd.DataFrame({f'{f}{k}': getattr(np, f)(2 * np.pi * k * doy / 365.25) for k in range(1, K + 1) for f in ('sin', 'cos')},
                        index=series.index)
X = fourier(2); four_name = 'ARIMA(2,0,1) + annual Fourier'
fou = SARIMAX(train, exog=X.iloc[:tr_end], order=(2, 0, 1), trend='c').fit(disp=False)
preds[four_name] = one_step(fou, X); OUT['aic'][four_name] = float(fou.aic)

classical = ['MA(7)', 'AR(14)', arima_name, sar_name, four_name]
for k_ in classical: results[k_] = metrics(test, preds[k_])
OUT['multi_step'] = multi
plot_forecast(preds['MA(7)'], 'MA(7): one-step-ahead forecast, test set', '03_ma_forecast.png')
plot_forecast(preds['AR(14)'], 'AR(14): one-step-ahead forecast, test set', '04_ar_forecast.png')
plot_forecast(preds[arima_name], f'{arima_name}: one-step-ahead forecast, test set', '05_arima_forecast.png')
plot_forecast(preds[four_name], f'{four_name}: one-step-ahead forecast, test set', '06_sarima_forecast.png')

# ---------------- neural models ----------------
scaler = MinMaxScaler().fit(train.values.reshape(-1, 1))
s_all = scaler.transform(series.values.reshape(-1, 1)).flatten()
inv = lambda a: scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).flatten()

def windows(lag, lo, hi):
    """Targets are days lo..hi-1; each input is the previous `lag` days (may reach back into the earlier set)."""
    Xs = np.stack([s_all[i - lag:i] for i in range(max(lo, lag), hi)])[..., None]
    return Xs, s_all[max(lo, lag):hi]

def build(kind, lag, units, layers, lr, loss='mse', dropout=0.0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    cell = SimpleRNN if kind == 'RNN' else LSTM
    m = Sequential([Input((lag, 1))])
    for i in range(layers): m.add(cell(units, return_sequences=(i < layers - 1)))
    if dropout: m.add(Dropout(dropout))
    m.add(Dense(1)); m.compile(optimizer=Adam(lr), loss=loss); return m

def fit(kind, lag, units, layers, lr, batch, loss='mse', dropout=0.0, epochs=40):
    Xt, yt = windows(lag, 0, tr_end); Xv, yv = windows(lag, tr_end, va_end)
    m = build(kind, lag, units, layers, lr, loss, dropout)
    h = m.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs, batch_size=batch, verbose=0,
              callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])
    vp = inv(m.predict(Xv, verbose=0))
    return m, float(np.sqrt(mean_squared_error(val.values, vp))), len(h.history['loss'])

tuning = {}
best_cfg = {}
for kind in ('RNN', 'LSTM'):
    rows = []
    for lag, units, lr in itertools.product((7, 14, 30), (32, 64), (0.001, 0.003)):
        _, v, ep = fit(kind, lag, units, 1, lr, 32)
        rows.append({'lag': lag, 'units': units, 'layers': 1, 'lr': lr, 'batch': 32, 'val_RMSE': v, 'epochs': ep})
        print(kind, rows[-1])
    b0 = min(rows, key=lambda r: r['val_RMSE'])
    for layers, batch in ((2, 32), (1, 64), (1, 16)):
        _, v, ep = fit(kind, b0['lag'], b0['units'], layers, b0['lr'], batch)
        rows.append({'lag': b0['lag'], 'units': b0['units'], 'layers': layers, 'lr': b0['lr'], 'batch': batch, 'val_RMSE': v, 'epochs': ep})
        print(kind, rows[-1])
    tuning[kind] = rows; best_cfg[kind] = min(rows, key=lambda r: r['val_RMSE'])
    c = best_cfg[kind]
    m, v, ep = fit(kind, c['lag'], c['units'], c['layers'], c['lr'], c['batch'])
    Xte, _ = windows(c['lag'], va_end, n)
    preds[kind] = inv(m.predict(Xte, verbose=0)); results[kind] = metrics(test, preds[kind])
OUT['tuning'] = tuning; OUT['best_cfg'] = best_cfg
plot_forecast(preds['RNN'], 'RNN: one-step-ahead forecast, test set', '07_rnn_forecast.png')
plot_forecast(preds['LSTM'], 'LSTM: one-step-ahead forecast, test set', '08_lstm_forecast.png')

# tuning chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, kind in zip(axes, ('RNN', 'LSTM')):
    g = pd.DataFrame(tuning[kind][:12])
    for (u, lr), c, ls in zip(itertools.product((32, 64), (0.001, 0.003)), (C_ACT, C_ACT, C_PRED, C_PRED), ('-', '--', '-', '--')):
        s = g[(g.units == u) & (g.lr == lr)]
        a.plot(s.lag.astype(str), s.val_RMSE, color=c, ls=ls, marker='o', label=f'{u} units, lr {lr}')
    a.set_title(f'{kind}: validation RMSE by lag window', loc='left'); a.set_xlabel('Lag window (days)')
axes[0].set_ylabel('Validation RMSE (°C)'); axes[1].legend()
plt.tight_layout(); plt.savefig('plots/08b_tuning.png'); plt.close()

# ---------------- quantile LSTMs (pinball loss) ----------------
def pinball(tau):
    def loss(y, p):
        e = y - p; return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss
qc = best_cfg['LSTM']; qpred = {}
Xte, _ = windows(qc['lag'], va_end, n); Xv, _ = windows(qc['lag'], tr_end, va_end)
qval = {}
for tau in (0.025, 0.10, 0.25, 0.40, 0.975):
    m, _, ep = fit('LSTM', qc['lag'], qc['units'], qc['layers'], qc['lr'], qc['batch'], loss=pinball(tau), epochs=60)
    qpred[tau] = inv(m.predict(Xte, verbose=0)); qval[tau] = inv(m.predict(Xv, verbose=0)); print('quantile', tau, 'epochs', ep)
lo, hi = qpred[0.025], qpred[0.975]; y = test.values
OUT['interval'] = {
    'PICP': float(np.mean((y >= lo) & (y <= hi))), 'MPIW': float(np.mean(hi - lo)),
    'below': int((y < lo).sum()), 'above': int((y > hi).sum()), 'crossing': int((lo > hi).sum()),
    'width_min': float((hi - lo).min()), 'width_max': float((hi - lo).max()),
    'max_breach_below': float(np.max(lo - y)), 'mean_breach_below': float(np.mean((lo - y)[y < lo])) if (y < lo).any() else 0.0,
    'PICP_val': float(np.mean((val.values >= qval[0.025]) & (val.values <= qval[0.975]))),
    'cov_below': {str(t): float(np.mean(y < qpred[t])) for t in qpred},
    'width_winter': float(np.mean((hi - lo)[np.isin(test.index.month, (6, 7, 8))])),
    'width_summer': float(np.mean((hi - lo)[np.isin(test.index.month, (12, 1, 2))])),
    'breach_dates_below': [str(d.date()) for d in test.index[y < lo]],
}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1.7, 1]})
axes[0].fill_between(test.index, lo, hi, color=C_BAND, alpha=0.28, label='95% prediction interval')
axes[0].plot(test.index, y, color=C_ACT, lw=0.9, label='Actual')
out = (y < lo) | (y > hi)
axes[0].scatter(test.index[out], y[out], color=C_PRED, s=22, zorder=5, label=f'Outside interval ({out.sum()} days)')
axes[0].set_title('95% prediction interval from two quantile LSTMs, test set', loc='left')
axes[0].set_ylabel('Temperature (°C)'); axes[0].legend(ncol=3, loc='upper center'); axes[0].set_ylim(-4, 28)
zi = (test.index >= '1990-06-01') & (test.index <= '1990-08-31')
axes[1].fill_between(test.index[zi], lo[zi], hi[zi], color=C_BAND, alpha=0.28)
axes[1].plot(test.index[zi], y[zi], color=C_ACT, lw=1.3, marker='o', ms=2.5)
axes[1].plot(test.index[zi], lo[zi], color=C_BAND, lw=1.2); axes[1].axhline(8, color='black', ls='--', lw=1)
axes[1].set_title('Zoom: winter 1990, with 8 °C threshold', loc='left'); axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.tight_layout(); plt.savefig('plots/09_prediction_interval.png'); plt.close()

# ---------------- decision quantification ----------------
THRESHOLD, COST, EMERG = 8.0, 12.0, 2.5
def cost(actual, plan_temp):
    need_a = np.maximum(THRESHOLD - actual, 0); need_f = np.maximum(THRESHOLD - plan_temp, 0)
    short = np.maximum(need_a - need_f, 0)
    return {'planned': float((need_f * COST).sum()), 'emergency': float((short * COST * EMERG).sum()),
            'total': float((need_f * COST + short * COST * EMERG).sum()),
            'emerg_nights': int((short > 0.05).sum()), 'short_deg': float(short.sum()),
            'wasted_deg': float(np.maximum(need_f - need_a, 0).sum())}
best_point = min(('RNN', 'LSTM'), key=lambda k_: results[k_]['RMSE'])
plans = {'No forecast (climatology)': preds['Climatology'], 'Persistence (yesterday)': preds['Persistence'],
         'AR(14)': preds['AR(14)'], arima_name: preds[arima_name], four_name: preds[four_name],
         'RNN point forecast': preds['RNN'], 'LSTM point forecast': preds['LSTM'],
         'Quantile LSTM, tau = 0.40': qpred[0.40], 'Quantile LSTM, tau = 0.25': qpred[0.25],
         'Quantile LSTM, tau = 0.10': qpred[0.10], 'Lower bound every night (tau = 0.025)': lo, 'Perfect foresight': y}
OUT['costs'] = {k_: cost(y, v) for k_, v in plans.items()}
OUT['best_point'] = best_point
OUT['need_deg_total'] = float(np.maximum(THRESHOLD - y, 0).sum())
# error profile of the best point model on cold nights
e = y - preds[best_point]
OUT['err'] = {'p95_abs': float(np.percentile(np.abs(e), 95)), 'max_over': float(e.min()), 'max_under': float(e.max()),
              'bias': float(e.mean()), 'cold_rmse': float(np.sqrt(np.mean(e[y < 8] ** 2))), 'cold_bias': float(e[y < 8].mean()),
              'warm_rmse': float(np.sqrt(np.mean(e[y >= 8] ** 2))),
              'worst_date': str(test.index[np.argmin(e)].date()), 'worst_actual': float(y[np.argmin(e)]), 'worst_pred': float(preds[best_point][np.argmin(e)])}

names = list(OUT['costs']); tot = [OUT['costs'][k_]['total'] for k_ in names]
pl = [OUT['costs'][k_]['planned'] for k_ in names]; em = [OUT['costs'][k_]['emergency'] for k_ in names]
fig, ax = plt.subplots(figsize=(12, 5.2)); yy = np.arange(len(names))[::-1]
ax.barh(yy, pl, color='#595959', label='Planned heating ($12 per °C)')
ax.barh(yy, em, left=pl, color=C_PRED, label='Emergency heating ($30 per °C)')
for i, t in zip(yy, tot): ax.text(t + 60, i, f'${t:,.0f}', va='center', color='black')
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlabel(f'Total heating cost over the {len(test)}-day test period (US$)')
ax.set_title('Heating cost by planning rule', loc='left'); ax.legend(loc='center right'); ax.grid(axis='y', visible=False)
ax.set_xlim(0, max(tot) * 1.13); plt.savefig('plots/10_cost_comparison.png'); plt.close()

order = ['Climatology', 'Persistence'] + classical + ['RNN', 'LSTM']
fig, ax = plt.subplots(figsize=(12, 4.4)); v = [results[k_]['RMSE'] for k_ in order]
bars = ax.bar(range(len(order)), v, color=['#a6a6a6', '#a6a6a6'] + ['#595959'] * len(classical) + [C_PRED, C_PRED])
for i, x in enumerate(v): ax.text(i, x + 0.05, f'{x:.2f}', ha='center', color='black')
ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace(' + ', '\n+ ').replace(')(', ')\n(') for o in order], fontsize=10)
ax.set_ylabel('Test RMSE (°C)'); ax.set_title('One-step-ahead test RMSE, all models', loc='left'); ax.grid(axis='x', visible=False)
plt.savefig('plots/11_rmse_comparison.png'); plt.close()

OUT['results'] = results; OUT['order'] = order
pd.DataFrame(results).T.loc[order].to_csv('results/model_metrics.csv')
pd.DataFrame(OUT['costs']).T.to_csv('results/decision_costs.csv')
pd.DataFrame({'Actual': y, **{k_: preds[k_] for k_ in order}, 'q0.025': lo, 'q0.10': qpred[0.10], 'q0.25': qpred[0.25], 'q0.40': qpred[0.40], 'q0.975': hi}, index=test.index).to_csv('results/test_predictions.csv')
json.dump(OUT, open('results/summary.json', 'w'), indent=1, default=str)
print(pd.DataFrame(results).T.loc[order].round(3)); print(json.dumps(OUT['interval'], indent=1)); print(pd.DataFrame(OUT['costs']).T.round(0))
print('Analysis complete.')
