"""Post-hoc analysis of the saved TPOT pipeline (NO new TPOT search): fingerprint-cluster-grouped CV of the fixed pipeline,
figures including TPOT, and permutation importance. Needs: outputs of Ge_X_BDE_revised.py and Ge_X_BDE_TPOT.py.
Run:  python Ge_X_BDE_TPOT_extras.py   (about 1 minute). Writes tpot_extras/."""
import os, numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
def find(name):
    for d in (".", "outputs", "tpot_outputs", "results", os.path.join("results", "main"), os.path.join("results", "tpot")):
        if os.path.exists(os.path.join(d, name)): return os.path.join(d, name)
    raise FileNotFoundError(name)
OUT = "tpot_extras"; os.makedirs(OUT, exist_ok=True); SEED = 42
D = pd.read_csv(find("GE_BDE_descriptors.csv")).merge(pd.read_csv(find("split_assignments.csv"))[["compound_id", "set"]], on="compound_id")
C = ["MolWt","LogP","TPSA","HBD","HBA","EN_diff","X_TotalValence","X_degree","Ge_degree","Ge_nb_C_count","Ge_nb_N_count","Ge_nb_O_count","Ge_nb_mean_EN","Ge_nb_max_EN","Shell2_N_count","Shell2_O_count","bond_Ge_C","bond_Ge_N","bond_Ge_O","Bond_Type"]
tr = D[D.set == "train"].reset_index(drop=True); X = tr[C].astype(float).values; y = tr.y.values
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.SMILES]
dist = []
for i in range(1, len(fp)): dist.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fp[i], fp[:i])])
cl = Butina.ClusterData(dist, len(fp), 0.6, isDistData=True); g = np.zeros(len(tr), int)
for ci, c in enumerate(cl):
    for j in c: g[j] = ci
pipe = joblib.load(find("tpot_best_pipeline.pkl"))
p = cross_val_predict(clone(pipe), X, y, cv=GroupKFold(5), groups=g)
res = {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** .5, "MAE": mean_absolute_error(y, p), "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}
pd.DataFrame([{"Model": "TPOT", "Evaluation": "CV cluster 5-fold (train; optimistic, pipeline selected on same data)", **res}]).round(4).to_csv(f"{OUT}/TPOT_cluster_CV.csv", index=False); print(res)
pd.DataFrame({"compound_id": tr.compound_id, "TPOT | cv_cluster": p}).to_csv(f"{OUT}/predictions_cv_cluster_train_TPOT.csv", index=False)
# permutation importance on TRAINING data only
pi = permutation_importance(pipe, X, y, n_repeats=10, random_state=SEED, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=C).sort_values(); imp.to_csv(f"{OUT}/Table_TPOT_permutation_importance_train.csv")
plt.figure(figsize=(7, 6)); imp.plot.barh(color="#4c72b0"); plt.xlabel("Increase in MAE when permuted (kcal/mol, training set)"); plt.title("TPOT pipeline: permutation importance"); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_TPOT_permutation_importance.png", dpi=300); plt.close()
# figures including TPOT
pe = pd.read_csv(find("predictions_external_test.csv")); pt = pd.read_csv(find("predictions_external_test_TPOT.csv")); pe["TPOT"] = pt.TPOT.values
pc = pd.read_csv(find("predictions_cv_train_TPOT.csv")).merge(tr[["compound_id", "y", "broken_bond"]], on="compound_id")
cols = {"Ge-C": "#2ca02c", "Ge-H": "#1f77b4", "Ge-N": "#ff7f0e", "Ge-O": "#d62728"}
def par(ax, yy, pp, b, title):
    for k, c in cols.items():
        m = (b == k).values; ax.scatter(yy[m], pp[m], s=14, c=c, label=k, alpha=.75)
    lim = [min(yy.min(), pp.min()) - 2, max(yy.max(), pp.max()) + 2]; ax.plot(lim, lim, "k--", lw=1)
    ax.set_title(f"{title}\nR²={r2_score(yy, pp):.3f}  MAE={mean_absolute_error(yy, pp):.2f}  MAPE={np.mean(abs((yy-pp)/yy))*100:.2f}%", fontsize=10)
    ax.set_xlabel("DFT De (kcal/mol)"); ax.set_ylabel("Predicted De (kcal/mol)")
yt = pe.y_true.values
fig, ax = plt.subplots(1, 4, figsize=(21, 5))
for a, n in zip(ax, ["TPOT", "XGBoost", "DNN (descriptor-based)", "RF"]): par(a, yt, pe[n].values, pe.broken_bond, n + " (external test)")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_parity_external_all_models_incl_TPOT.png", dpi=300); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
par(ax[0], pc.y.values, pc["TPOT | cv_random"].values, pc.broken_bond, "TPOT 5-fold CV (train; optimistic)"); par(ax[1], yt, pe.TPOT.values, pe.broken_bond, "TPOT external test")
r_ = pe.TPOT.values - yt; ax[2].scatter(yt, r_, s=14, c=[cols[b] for b in pe.broken_bond], alpha=.75); ax[2].axhline(0, c="k", ls="--", lw=1)
ax[2].set_xlabel("DFT De (kcal/mol)"); ax[2].set_ylabel("Residual (pred - DFT, kcal/mol)"); ax[2].set_title(f"TPOT external residuals (mean {r_.mean():+.2f}, SD {r_.std():.2f})", fontsize=10)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_TPOT_CV_external_residuals.png", dpi=300); plt.close()
perf = pd.read_csv(find("Table_model_performance.csv")); tp = pd.read_csv(find("TPOT_results.csv")); rows = []
for n in ["XGBoost", "RF", "DNN (descriptor-based)"]:
    for e in ["CV random", "CV cluster", "External"]: rows.append((n, e, perf[(perf.Model == n) & perf.Evaluation.str.startswith(e)].MAE.iloc[0]))
rows += [("TPOT", "CV random", tp.MAE.iloc[0]), ("TPOT", "CV cluster", res["MAE"]), ("TPOT", "External", tp.MAE.iloc[1])]
df = pd.DataFrame(rows, columns=["Model", "Eval", "MAE"]).pivot(index="Model", columns="Eval", values="MAE")[["CV random", "CV cluster", "External"]]
ax = df.plot.barh(figsize=(10, 5)); ax.set_xlabel("MAE (kcal/mol)"); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_MAE_comparison_incl_TPOT.png", dpi=300); plt.close(); print(df.round(3))
