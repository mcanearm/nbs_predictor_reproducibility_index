"""
Script 3: Results Compilation and Figure Generation

Loads pre-computed cross-validation results from data/model_results/ and
generates all tables and figures used in the paper.

Inputs:
  - data/model_results/{model}_{split}.csv
  - data/models/VARX_9.pkl                  (for correlation / trace plots)
  - data/y_scaler.pkl

Outputs (all written to images/):
  - images/model_comparison.pdf
  - images/ar_comparison.pdf
  - images/correlation_matrices.pdf

Run from the project root:
    python scripts/03_results_and_plots.py
"""

import pickle as pkl
import sys
from functools import partial, reduce
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.modeling.metrics import summarize

RESULTS_DIR = Path("data/model_results")
MODEL_DIR = Path("data/models")
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)


def rename_lakes(x):
    return {
        "eri": "Erie",
        "sup": "Superior",
        "mic_hur": "Michigan/Huron",
        "ont": "Ontario",
    }.get(x, x)


# --- Load y_scaler ---
with open(Path("data") / "y_scaler.pkl", "rb") as f:
    y_scaler = pkl.load(f)

# --- Load all CV result CSVs ---
result_files = sorted(RESULTS_DIR.glob("*.csv"))
if not result_files:
    raise FileNotFoundError(
        f"No result CSVs found in {RESULTS_DIR}. Run 02_fit_models.py first."
    )

cv_results = pd.concat(
    [pd.read_csv(f) for f in result_files], axis=0, ignore_index=True
)
cv_results["Date"] = pd.to_datetime(cv_results["Date"])

# --- Process results into scaled_data and unscaled_data ---
cv_results = cv_results[cv_results["months_ahead"] <= 12]
unscaled_predictions = cv_results[
    [
        "Date",
        "lake",
        "model",
        "mean",
        "lower",
        "upper",
        "true",
        "std",
        "months_ahead",
        "split",
    ]
].melt(id_vars=["Date", "model", "lake", "months_ahead", "split"])

unscaled_predictions["lake"] = pd.Categorical(
    unscaled_predictions["lake"],
    categories=["sup", "mic_hur", "eri", "ont"],
    ordered=True,
)

grped_data = (
    unscaled_predictions.pivot_table(
        index=["Date", "model", "variable", "months_ahead", "split"],
        columns=["lake"],
        values="value",
    )
    .reset_index()
    .groupby("variable")
)

scaled_data = []
unscaled_data = []
for var_name, df in grped_data:
    scaled_sub_df = df.copy()
    df[["sup", "mic_hur", "eri", "ont"]] = y_scaler.inverse_transform(
        df[["sup", "mic_hur", "eri", "ont"]]
    )
    df = df.melt(
        id_vars=["Date", "model", "months_ahead", "split"],
        value_vars=["sup", "mic_hur", "eri", "ont"],
        var_name="lake",
        value_name=var_name,
    ).set_index(["Date", "model", "lake", "months_ahead", "split"])
    unscaled_data.append(df)
    scaled_sub_df = scaled_sub_df.melt(
        id_vars=["Date", "model", "months_ahead", "split"],
        value_vars=["sup", "mic_hur", "eri", "ont"],
        var_name="lake",
        value_name=var_name,
    ).set_index(["Date", "model", "lake", "months_ahead", "split"])
    scaled_data.append(scaled_sub_df)

scaled_data = reduce(
    partial(pd.merge, left_index=True, right_index=True), scaled_data
).reset_index()
unscaled_data = reduce(
    partial(pd.merge, left_index=True, right_index=True), unscaled_data
).reset_index()

# --- Table: Per-lake performance ---
print("\n=== Per-lake performance ===")
print(
    scaled_data.groupby(["lake", "model"])
    .apply(summarize, include_groups=False)
    .reset_index()
    .to_string()
)

# --- Table: Overall performance ---
print("\n=== Overall model performance ===")
print(
    scaled_data.groupby(["model"])
    .apply(summarize, include_groups=False)
    .round(3)
    .to_string()
)


# --- Table: CRPS (LaTeX) ---
def rollup_agg(df, groupings, fn=summarize):
    full_groupings = (
        df.groupby(groupings, observed=False)
        .apply(fn, include_groups=False)
        .reset_index()
    )
    all_groupings = (
        df.groupby(groupings[:-1], observed=False)
        .apply(fn, include_groups=False)
        .reset_index()
    )
    all_groupings[groupings[-1]] = "All"
    return pd.concat(
        [full_groupings, all_groupings.assign(lake="All")], axis=0, ignore_index=True
    ).set_index(groupings)


CRPS_table = rollup_agg(scaled_data, ["model", "lake"])["crps"].reset_index()
rename_lakes_vec = np.vectorize(rename_lakes)
CRPS_table["lake"] = pd.Categorical(
    rename_lakes_vec(CRPS_table["lake"]),
    categories=["Superior", "Michigan/Huron", "Erie", "Ontario", "All"],
)
print("\n=== CRPS table (LaTeX) ===")
print(
    CRPS_table.pivot_table(columns=["lake"], index=["model"], values="crps").to_latex(
        float_format="%.3f"
    )
)

# --- Table: Performance by months-ahead (informational) ---
# NOTE: with only 10 splits this table has limited statistical power
print("\n=== Performance by months ahead ===")
print(
    scaled_data[scaled_data["months_ahead"].isin([1, 3, 6, 12])]
    .groupby(["months_ahead", "model"])
    .apply(summarize, include_groups=False)
    .reset_index()
    .to_string()
)

# --- Figure: Model forecast comparison ---
LAKE_ORDER = ["sup", "mic_hur", "eri", "ont"]
MODEL_ORDER = ["Default", "GP_Matern", "VARX", "SimpleLM", "NARX"]
LABEL_MAP = {
    "sup": "Superior",
    "mic_hur": "Michigan/Huron",
    "eri": "Erie",
    "ont": "Ontario",
    "GP_Matern": "GP",
}


def facet_rename(x):
    return LABEL_MAP.get(x, x)


unscaled_plot_data = scaled_data[
    (scaled_data["model"].isin(MODEL_ORDER)) & (scaled_data["split"] >= 6)
].copy()

models_present = [m for m in MODEL_ORDER if m in unscaled_plot_data["model"].unique()]
lakes_present = [l for l in LAKE_ORDER if l in unscaled_plot_data["lake"].unique()]

fig, axs = plt.subplots(
    len(models_present),
    len(lakes_present),
    figsize=(12, 6),
    sharex=True,
    sharey="row",
)
# ensure 2-D even with a single row/col
axs = np.atleast_2d(axs)

for row, model in enumerate(models_present):
    for col, lake in enumerate(lakes_present):
        ax = axs[row, col]
        subset = unscaled_plot_data[
            (unscaled_plot_data["model"] == model)
            & (unscaled_plot_data["lake"] == lake)
        ].sort_values("Date")

        ax.plot(subset["Date"], subset["true"], color="black", lw=0.8, label="Observed")
        ax.plot(subset["Date"], subset["mean"], color="red", lw=0.8, label="Forecast")
        ax.fill_between(
            subset["Date"],
            subset["lower"],
            subset["upper"],
            color="red",
            alpha=0.25,
            label="95% CI",
        )

        if row == 0:
            ax.set_title(facet_rename(lake), fontsize=11)
        if col == 0:
            ax.set_ylabel(facet_rename(model), fontsize=10)

        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

# shared y-axis label
fig.text(0.04, 0.5, r"RNBS $(m^3/s)$", va="center", rotation="vertical", fontsize=12)

# single legend from the first subplot
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0.05, 0.05, 1, 1])
fig.savefig(IMAGES_DIR / "model_comparison.pdf")
plt.close(fig)
print(f"\nSaved {IMAGES_DIR / 'model_comparison.pdf'}")

# --- Figure: Autoregressive term comparison ---
scaled_ar_data = scaled_data[
    scaled_data["model"].isin(["VARX", "NARX", "VARX_lag2", "VARX_lag3"])
]
ar_plot_data = (
    scaled_ar_data.groupby(["months_ahead", "model", "lake"])
    .apply(summarize, include_groups=False)
    .reset_index()
)

lake_list = scaled_ar_data["lake"].unique().tolist()
nrows, ncols = 2, 2

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10), sharey=True)
axs = axs.flatten()

for i, lake in enumerate(lake_list):
    ax = axs[i]
    for model, group in ar_plot_data[ar_plot_data["lake"] == lake].groupby("model"):
        ax.plot(group["months_ahead"], group["rmse"], label=model, marker="o")
    ax.set_title(facet_rename(lake))
    ax.set_xticks(group["months_ahead"].unique())
    ax.set_xticklabels(group["months_ahead"].unique())

for j in range(i + 1, nrows * ncols):
    fig.delaxes(axs[j])

fig.text(0.5, 0.04, "Months Ahead", ha="center", va="center", fontsize=14)
fig.text(0.03, 0.5, "RMSE", ha="center", va="center", rotation="vertical", fontsize=14)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0.04, 0.06, 1, 0.97])
plt.savefig(str(IMAGES_DIR / "ar_comparison.pdf"))
plt.close()
print(f"Saved {IMAGES_DIR / 'ar_comparison.pdf'}")

# --- Figure: VARX correlation matrix (posterior) ---
varx_model_path = MODEL_DIR / "VARX_9.pkl"
if not varx_model_path.exists():
    print(f"\nSkipping correlation/trace plots: {varx_model_path} not found.")
else:
    with open(varx_model_path, "rb") as f:
        varx_model = pkl.load(f)
    trace = varx_model["model"].trace_

    valid_set = [
        ("sup", "sup"),
        ("mic_hur", "sup"),
        ("mic_hur", "mic_hur"),
        ("eri", "sup"),
        ("eri", "mic_hur"),
        ("eri", "eri"),
        ("ont", "sup"),
        ("ont", "mic_hur"),
        ("ont", "eri"),
        ("ont", "ont"),
    ]

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(6, 6))
    axs = axs.flatten()

    triangle_plots = [0, 4, 5, 8, 9, 10, 12, 13, 14, 15]
    valid_pairs = valid_set.copy()

    for idx in range(16):
        ax = axs[idx]
        ax.set_prop_cycle(None)

        if idx not in triangle_plots:
            ax.set_visible(False)
            continue

        series, lake = valid_pairs.pop(0)
        subtrace = trace.posterior["corr"].sel(series=series, lakes=lake)

        for c in range(subtrace.chain.size):
            sns.kdeplot(
                subtrace.sel(chain=subtrace.chain.values[c]).values.flatten(),
                ax=ax,
                fill=False,
                alpha=0.5,
                linewidth=1,
            )

        ax.set_xlim(-1.25, 1.25)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

        row = idx // 4
        col = idx % 4

        if row == 3:
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_xticklabels(
                ["–1.0", "–0.5", "0", "0.5", "1.0"], ha="center", fontsize=8
            )
        else:
            ax.set_xticks([])
            ax.tick_params(axis="x", labelsize=0)

        if row == col:
            ax.set_title(rename_lakes(lake), fontsize=10, pad=6)

        if col == 0:
            ax.annotate(
                rename_lakes(series),
                xy=(0, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=10,
                xytext=(-6, 0),
                textcoords="offset points",
            )

    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.20, right=0.95, top=0.95, bottom=0.08
    )
    plt.savefig(str(IMAGES_DIR / "correlation_matrices.pdf"))
    plt.close()
    print(f"Saved {IMAGES_DIR / 'correlation_matrices.pdf'}")

    az.plot_trace(trace, var_names=["corr", "intercept"], compact=True)
    plt.tight_layout()
    plt.savefig(str(IMAGES_DIR / "varx_trace.pdf"))
    plt.close()
    print(f"Saved {IMAGES_DIR / 'varx_trace.pdf'}")
