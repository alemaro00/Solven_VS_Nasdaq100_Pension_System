from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
BOOK2_PATH = DATA_DIR / "Book2.csv"


def risk_metrics(price: pd.Series, rf_ann: float = 0.0) -> dict[str, float]:
    """Calcola metriche rischio-rendimento usando rendimenti mensili semplici."""
    rets = price.pct_change().dropna()
    if rets.empty:
        return {
            "CAGR": np.nan,
            "Vol_ann": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "MaxDD": np.nan,
            "Calmar": np.nan,
        }

    n_years = (price.index[-1] - price.index[0]).days / 365.25
    cagr = (price.iloc[-1] / price.iloc[0]) ** (1 / n_years) - 1

    vol_ann = rets.std(ddof=1) * np.sqrt(12)
    rf_month = (1 + rf_ann) ** (1 / 12) - 1
    excess = rets - rf_month

    sharpe = (excess.mean() / rets.std(ddof=1)) * np.sqrt(12) if rets.std(ddof=1) > 0 else np.nan

    downside = np.minimum(excess, 0.0)
    downside_dev_ann = np.sqrt((downside**2).mean()) * np.sqrt(12)
    sortino = ((excess.mean() * 12) / downside_dev_ann) if downside_dev_ann > 0 else np.nan

    wealth = price / price.iloc[0]
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "CAGR": cagr,
        "Vol_ann": vol_ann,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
    }


# ── 1. carica Book2.csv ────────────────────────────────────────────────────────
book = pd.read_csv(BOOK2_PATH, sep=";", dtype=str)
# pulizia: alcune righe usano ',' come separatore decimale, altre '.'
for col in ["Reddito", "Dinamico"]:
    book[col] = book[col].str.replace(",", ".").astype(float)
book["Data"] = pd.to_datetime(book["Data"], dayfirst=True)
book = book.sort_values("Data").set_index("Data")
book.index = book.index.to_period("M").to_timestamp("M")   # fine mese canonico

print(f"Book2.csv caricato: {len(book)} osservazioni")
print(f"  Periodo: {book.index[0].date()} → {book.index[-1].date()}")
print()

# ── 2. scarica indici da yfinance ─────────────────────────────────────────────
start = book.index[0] - pd.offsets.MonthBegin(1)
end   = book.index[-1] + pd.offsets.MonthEnd(1)

print(f"Scarico S&P500 e Nasdaq100 da {start.date()} a {end.date()} …")
raw = yf.download(["^GSPC", "^NDX"], start=start, end=end,
                  auto_adjust=True, progress=False)["Close"]
raw.index = pd.to_datetime(raw.index)

# resample mensile: ultimo giorno disponibile del mese
monthly = raw.resample("ME").last()
monthly.index = monthly.index.to_period("M").to_timestamp("M")  # allinea a fine mese

# allinea sulle stesse date del book
df = book.join(monthly, how="inner")
df.columns = ["Reddito", "Dinamico", "SP500", "NDX100"]
df = df.dropna()

print(f"Osservazioni dopo join: {len(df)}")
print()
print(df.round(3).to_string())
print()

# ── 3. grafici comparativi ────────────────────────────────────────────────────
print("=" * 60)
print("GRAFICI COMPARATIVI")
print("=" * 60)

plot_dir = DATA_DIR

# Livelli normalizzati a base 100 per confronto diretto tra serie con scale diverse.
base100 = df.div(df.iloc[0]).mul(100)
fig, ax = plt.subplots(figsize=(12, 7))
for col in ["Reddito", "Dinamico", "SP500", "NDX100"]:
    ax.plot(base100.index, base100[col], label=col, linewidth=2)
ax.set_title("Confronto Livelli Normalizzati (Base 100)")
ax.set_xlabel("Data")
ax.set_ylabel("Indice (Base=100)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plot1_path = plot_dir / "confronto_livelli_normalizzati.png"
fig.savefig(plot1_path, dpi=150)
plt.close(fig)

print(f"Grafico salvato: {plot1_path}")
print()

# ── 4. rendimenti annuali ─────────────────────────────────────────────────────
print("=" * 60)
print("RENDIMENTI ANNUALI (quota fine anno / quota fine anno precedente - 1)")
print("=" * 60)

# Prendo il valore di fine anno: l'ultima quota disponibile per ogni anno.
# Per l'anno corrente (incompleto) uso l'ultima quota disponibile.
annual = df.groupby(df.index.year).last()
annual_ret = annual.pct_change().dropna()
annual_ret.index.name = "Anno"

# formattazione in %
annual_ret_pct = (annual_ret * 100).round(2)
annual_ret_pct.columns = ["Reddito %", "Dinamico %", "SP500 %", "NDX100 %"]
print(annual_ret_pct.to_string())
print()

# statistiche riassuntive sui rendimenti annuali
print("Statistiche sui rendimenti annuali (%):")
print(annual_ret_pct.describe().round(2).to_string())
print()

# rendimento medio annuo composto (CAGR) sull'intero periodo
n_anni = (df.index[-1] - df.index[0]).days / 365.25
print(f"CAGR sull'intero periodo ({df.index[0].year}–{df.index[-1].year}, {n_anni:.1f} anni):")
for col, label in [("Reddito", "Reddito"), ("Dinamico", "Dinamico"),
                   ("SP500", "SP500  "), ("NDX100", "NDX100 ")]:
    cagr = (df[col].iloc[-1] / df[col].iloc[0]) ** (1 / n_anni) - 1
    print(f"  {label}: {cagr * 100:.2f}% annuo")
print()

# ── 5. metriche rischio-rendimento ────────────────────────────────────────────
print("=" * 60)
print("METRICHE RISCHIO-RENDIMENTO")
print("=" * 60)
print("Assunzione: tasso risk-free annuo = 0.00%")
print()

metrics = pd.DataFrame({col: risk_metrics(df[col], rf_ann=0.0) for col in df.columns}).T
metrics_pct = metrics.copy()
for col in ["CAGR", "Vol_ann", "MaxDD"]:
    metrics_pct[col] = metrics_pct[col] * 100

metrics_pct = metrics_pct.rename(columns={
    "CAGR": "CAGR %",
    "Vol_ann": "Volatilita annua %",
    "Sharpe": "Sharpe",
    "Sortino": "Sortino",
    "MaxDD": "Max Drawdown %",
    "Calmar": "Calmar",
})

print(metrics_pct.round(3).to_string())
print()

# ── 6. correlazioni di Pearson ────────────────────────────────────────────────
print("=" * 60)
print("CORRELAZIONI DI PEARSON (livelli)")
print("=" * 60)
corr_matrix = df.corr(method="pearson")
print(corr_matrix.round(4).to_string())
print()

# correlazioni con p-value
pairs = [
    ("Reddito", "SP500"),
    ("Reddito", "NDX100"),
    ("Dinamico", "SP500"),
    ("Dinamico", "NDX100"),
]
print("Correlazioni con p-value:")
for a, b in pairs:
    r, p = stats.pearsonr(df[a], df[b])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {a} ~ {b}: r={r:.4f}  p={p:.4f} {sig}")
print()

# correlazioni sui rendimenti mensili (log-returns) — più robusti
print("=" * 60)
print("CORRELAZIONI DI PEARSON (log-rendimenti mensili)")
print("=" * 60)
returns = np.log(df).diff().dropna()
ret_corr = returns.corr(method="pearson")
print(ret_corr.round(4).to_string())
print()

print("Correlazioni rendimenti con p-value:")
for a, b in pairs:
    r, p = stats.pearsonr(returns[a], returns[b])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {a} ~ {b}: r={r:.4f}  p={p:.4f} {sig}")
print()

# ── 7. test di correlazione robusti (Spearman, Kendall) ──────────────────────
print("=" * 60)
print("CORRELAZIONI ROBUSTE - SPEARMAN RANK (sui livelli)")
print("=" * 60)
print("Test non parametrico: misura correlazione monotonica (non lineare).")
print("Più robusto a outlier rispetto a Pearson.")
print()
for a, b in pairs:
    rho, p = stats.spearmanr(df[a], df[b])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {a} ~ {b}: ρ={rho:.4f}  p={p:.4f} {sig}")
print()

print("=" * 60)
print("CORRELAZIONI ROBUSTE - KENDALL TAU (sui livelli)")
print("=" * 60)
print("Test non parametrico: basato su concordanza di ranghi.")
print("Ancora più robusto a outlier, interpretazione probabilistica.")
print()
for a, b in pairs:
    tau, p = stats.kendalltau(df[a], df[b])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {a} ~ {b}: τ={tau:.4f}  p={p:.4f} {sig}")
print()

# Confronto tra i tre metodi di correlazione
print("=" * 60)
print("CONFRONTO TRE METODI DI CORRELAZIONE (sui livelli)")
print("=" * 60)
print(f"{'Coppia':<25} {'Pearson':<12} {'Spearman':<12} {'Kendall':<12}")
print("-" * 60)
for a, b in pairs:
    pearson, _ = stats.pearsonr(df[a], df[b])
    spearman, _ = stats.spearmanr(df[a], df[b])
    kendall, _ = stats.kendalltau(df[a], df[b])
    print(f"{a} ~ {b:<16} {pearson:>11.4f}  {spearman:>11.4f}  {kendall:>11.4f}")
print()
