# %% [markdown]
# # Ge-X electronic dissociation energy (De) prediction: leakage-safe, reproducible workflow
# **Inputs (upload to the Colab working folder):** `GE_BDE.csv` (raw: compound_id, SMILES, broken_bond, BDE_EE_kcal_mol) and `Ge-Test-Dataset.csv` (used ONLY to read which SMILES form the published external test set).
# Everything else (descriptors, split file, tables, figures) is produced here. Run: Runtime > Restart and run all.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit', 'xgboost', 'shap', 'scikit-learn', 'matplotlib'])

# %%
import os, sys, json, random, warnings, time, platform
import importlib.metadata as md
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---------------- CONFIG (the only cell you should edit) ----------------
SEED          = 42
RAW_CSV       = "GE_BDE.csv"
TEST_LIST_CSV = "Ge-Test-Dataset.csv"      # defines the published 183-compound external test set
OUT           = "outputs"
FEATURE_MODE  = "curated"   # "curated": your 19 descriptors + Bond_Type (nothing fitted on data)
                            # "auto"   : all 32 descriptors + Bond_Type, filter REFIT inside every training fold
PAIR_THR, TARGET_THR = 0.85, 0.05           # used only when FEATURE_MODE == "auto"
QUICK         = bool(int(os.environ.get("QUICK", "0")))   # 1 = smoke test only
DNN_EPOCHS    = 15 if QUICK else 400
RUN_TPOT      = False      # set True only after the other cells ran; slow
# -------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
pk = ["numpy","pandas","scikit-learn","xgboost","rdkit","shap","matplotlib","scipy"]
VERS = {p: (md.version(p) if p in {d.metadata["Name"].lower() for d in md.distributions()} or True else "") for p in pk}
for p in list(VERS):
    try: VERS[p] = md.version(p)
    except Exception: VERS[p] = "n/a"
VERS["python"] = sys.version.split()[0]
print(json.dumps(VERS, indent=1))

# %% [markdown]
# ## 1. Descriptors (RDKit) from the raw file

# %%
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
EN = {"H":2.20,"C":2.55,"N":3.04,"O":3.44,"Ge":2.01,
      "Si":1.90,"P":2.19,"S":2.58,"F":3.98,"Cl":3.16,
      "Br":2.96,"I":2.66,"B":2.04,"Se":2.55}

BOND_TYPES = ["Ge-H","Ge-C","Ge-N","Ge-O"]
BOND_TO_X  = {"Ge-H":"H","Ge-C":"C","Ge-N":"N","Ge-O":"O"}


def get_descriptors(smi, bond_label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        ComputeGasteigerCharges(mol)
    except Exception:
        pass

    feats = {}

    # GROUP 1 — Physicochemical
    feats["MolWt"]            = round(Descriptors.MolWt(mol), 4)
    feats["LogP"]             = round(Descriptors.MolLogP(mol), 4)
    feats["TPSA"]             = round(Descriptors.TPSA(mol), 4)
    feats["HBD"]              = rdMolDescriptors.CalcNumHBD(mol)
    feats["HBA"]              = rdMolDescriptors.CalcNumHBA(mol)

    # GROUP 2 — Structural
    feats["NumRotBonds"]      = rdMolDescriptors.CalcNumRotatableBonds(mol)
    feats["NumRings"]         = rdMolDescriptors.CalcNumRings(mol)
    feats["NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    feats["FractionCsp3"]     = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)

    # Find Ge and X atoms
    x_sym   = BOND_TO_X.get(bond_label, "H")
    ge_atom = None
    x_atom  = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Ge":
            ge_atom = atom
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() == x_sym and x_atom is None:
                    x_atom = nb
            break
    if ge_atom is None:
        return None

    # GROUP 3 — Ge-X Bond
    if x_atom is None and x_sym == "H":
        feats["X_AtomicNum"]     = 1
        feats["X_mass"]          = 1.008
        feats["EN_diff"]         = round(abs(EN["Ge"] - EN["H"]), 4)
        feats["Ge_X_EN_product"] = round(EN["Ge"] * EN["H"], 4)
        feats["X_TotalValence"]  = 1
        feats["X_degree"]        = 0
        feats["X_hybridization"] = 0
        feats["X_is_aromatic"]   = 0
    elif x_atom is not None:
        sym = x_atom.GetSymbol()
        feats["X_AtomicNum"]     = x_atom.GetAtomicNum()
        feats["X_mass"]          = round(x_atom.GetMass(), 4)
        feats["EN_diff"]         = round(abs(EN.get("Ge",2.01)-EN.get(sym,2.0)), 4)
        feats["Ge_X_EN_product"] = round(EN.get("Ge",2.01)*EN.get(sym,2.0), 4)
        feats["X_TotalValence"]  = x_atom.GetTotalValence()
        feats["X_degree"]        = x_atom.GetDegree()
        feats["X_hybridization"] = int(x_atom.GetHybridization())
        feats["X_is_aromatic"]   = int(x_atom.GetIsAromatic())
    else:
        for k in ["X_AtomicNum","X_mass","EN_diff","Ge_X_EN_product",
                  "X_TotalValence","X_degree","X_hybridization","X_is_aromatic"]:
            feats[k] = 0.0

    # GROUP 4 — Ge Local Environment
    ge_neighbors = ge_atom.GetNeighbors()
    nb_syms = [nb.GetSymbol() for nb in ge_neighbors]
    nb_ens  = [EN.get(s, 2.0) for s in nb_syms]

    feats["Ge_degree"]     = ge_atom.GetDegree()
    feats["Ge_attached_H"] = ge_atom.GetTotalNumHs()
    feats["Ge_nb_C_count"] = nb_syms.count("C")
    feats["Ge_nb_N_count"] = nb_syms.count("N")
    feats["Ge_nb_O_count"] = nb_syms.count("O")
    feats["Ge_nb_mean_EN"] = round(float(np.mean(nb_ens)), 4) if nb_ens else 0.0
    feats["Ge_nb_max_EN"]  = round(float(np.max(nb_ens)),  4) if nb_ens else 0.0

    # GROUP 5 — Second Shell
    shell2 = [nb2.GetSymbol()
              for nb in ge_neighbors
              for nb2 in nb.GetNeighbors()
              if nb2.GetIdx() != ge_atom.GetIdx()]

    feats["Shell2_count"]   = len(shell2)
    feats["Shell2_mean_EN"] = round(float(np.mean([EN.get(s,2.0) for s in shell2])), 4) if shell2 else 0.0
    feats["Shell2_N_count"] = shell2.count("N")
    feats["Shell2_O_count"] = shell2.count("O")

    # GROUP 6 — Bond-type one-hot (4 types only)
    for bt in BOND_TYPES:
        feats[f"bond_{bt.replace('-','_')}"] = int(bond_label == bt)

    return feats




raw = pd.read_csv(RAW_CSV)
rows = []
for _, r in raw.iterrows():
    d = get_descriptors(r["SMILES"], r["broken_bond"])
    if d is None: raise ValueError(f"descriptor failure: {r['compound_id']}")
    rows.append({"compound_id": r["compound_id"], "SMILES": r["SMILES"], "broken_bond": r["broken_bond"],
                 "y": r["BDE_EE_kcal_mol"], **d})
D = pd.DataFrame(rows)
BOND_CODE = {"Ge-C": 0, "Ge-H": 1, "Ge-N": 2, "Ge-O": 3}     # fixed, documented (ordinal label, not physical order)
D["Bond_Type"] = D["broken_bond"].map(BOND_CODE)
META = ["compound_id", "SMILES", "broken_bond", "y"]
ALL_DESC = [c for c in D.columns if c not in META]            # 32 RDKit descriptors + Bond_Type
CURATED = ["MolWt","LogP","TPSA","HBD","HBA","EN_diff","X_TotalValence","X_degree","Ge_degree",
           "Ge_nb_C_count","Ge_nb_N_count","Ge_nb_O_count","Ge_nb_mean_EN","Ge_nb_max_EN",
           "Shell2_N_count","Shell2_O_count","bond_Ge_C","bond_Ge_N","bond_Ge_O","Bond_Type"]
D.to_csv(f"{OUT}/GE_BDE_descriptors.csv", index=False)
print(D.shape, "| model inputs in curated mode:", len(CURATED), "| candidate descriptors:", len(ALL_DESC))

# %% [markdown]
# ## 2. Deterministic, auditable split + chemistry-grouped folds
# The published split is reproduced from `Ge-Test-Dataset.csv`; `split_assignments.csv` records it. `regenerate_split()` shows how a new stratified split would be drawn from the seed.

# %%
from sklearn.model_selection import train_test_split, KFold, GroupKFold
test_smiles = set(pd.read_csv(TEST_LIST_CSV)["SMILES"])
D["set"] = np.where(D["SMILES"].isin(test_smiles), "test", "train")
assert (D.set == "test").sum() == len(test_smiles) == 183 and (D.set == "train").sum() == 1000
D[["compound_id","SMILES","broken_bond","set"]].to_csv(f"{OUT}/split_assignments.csv", index=False)

def regenerate_split(df, n_test=183, seed=SEED):          # not used for the paper; for documentation
    _, te = train_test_split(df.index, test_size=n_test, stratify=df["broken_bond"], random_state=seed)
    return np.where(df.index.isin(te), "test", "train")

tr = D[D.set == "train"].reset_index(drop=True); te = D[D.set == "test"].reset_index(drop=True)
ytr, yte = tr.y.values, te.y.values
print(tr.broken_bond.value_counts(normalize=True).round(3).to_dict()); print(te.broken_bond.value_counts(normalize=True).round(3).to_dict())

# Morgan fingerprints, Butina clusters (training set only), nearest-neighbour similarity of test to train
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp_tr = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.SMILES]
fp_te = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in te.SMILES]
dist = []
for i in range(1, len(fp_tr)):
    dist.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fp_tr[i], fp_tr[:i])])
clusters = Butina.ClusterData(dist, len(fp_tr), 0.6, isDistData=True)
groups = np.zeros(len(tr), int)
for ci, c in enumerate(clusters):
    for j in c: groups[j] = ci
nn_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fp_tr)) for f in fp_te])
sim_summary = {"n_clusters": len(clusters), "test_nn_tanimoto_median": float(np.median(nn_sim)),
               "test_nn_tanimoto_mean": float(nn_sim.mean()), "n_test_gt_0.7": int((nn_sim > 0.7).sum()),
               "n_test_gt_0.8": int((nn_sim > 0.8).sum())}
print(sim_summary)

key = lambda d: d[CURATED[:-1]].round(4).astype(str).agg("|".join, axis=1) + "|" + d.broken_bond
dup_mask = key(te).isin(set(key(tr))).values
print("test compounds whose 19-descriptor vector (+bond) equals a training compound:", int(dup_mask.sum()))
RANDOM_CV  = KFold(5, shuffle=True, random_state=SEED)
CLUSTER_CV = GroupKFold(5)

# %% [markdown]
# ## 3. Leakage-safe pipeline pieces (everything target-dependent is fitted inside the training fold)

# %%
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Fixed column list (nothing is learned from data)."""
    def __init__(self, cols=None): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[list(self.cols)]

class FeatureReducer(BaseEstimator, TransformerMixin):
    """Constant removal -> |r| with target filter -> pairwise |r| filter. Fitted on the training fold only."""
    def __init__(self, cols=None, target_thr=0.05, pair_thr=0.85, protect=("Bond_Type",)):
        self.cols, self.target_thr, self.pair_thr, self.protect = cols, target_thr, pair_thr, protect
    def fit(self, X, y):
        X = X[list(self.cols)].astype(float); y = np.asarray(y, float)
        X = X.loc[:, X.std() > 0]
        t = {c: abs(np.corrcoef(X[c], y)[0, 1]) for c in X}
        X = X[[c for c in X if t[c] >= self.target_thr or c in self.protect]]
        cm = X.corr().abs(); up = cm.where(np.triu(np.ones(cm.shape, bool), k=1)); drop = set()
        for col in up.columns:
            for p in up[col][up[col] > self.pair_thr].index:
                if col in drop or p in drop: continue
                drop.add(p if (t[col] >= t[p] or p not in self.protect and col in self.protect) else col)
        self.keep_ = [c for c in X if c not in drop]; return self
    def transform(self, X): return X[self.keep_]

def selector(mode=None, drop=()):
    mode = mode or FEATURE_MODE
    if mode == "curated": return ColumnSelector([c for c in CURATED if c not in drop])
    return FeatureReducer([c for c in ALL_DESC if c not in drop], TARGET_THR, PAIR_THR)

class DescriptorDNN(BaseEstimator, RegressorMixin):
    """Feed-forward network on ONE descriptor vector per molecule (NOT a message-passing/graph model).
    d -> 5 x [Dense(64)+LeakyReLU] -> Dense(1); He init, Adam, MSE, lr = lr0 * decay**epoch; NumPy only."""
    def __init__(self, width=64, n_hidden=5, epochs=400, batch=64, lr=5e-4, decay=0.997, seed=42):
        self.width, self.n_hidden, self.epochs, self.batch, self.lr, self.decay, self.seed = width, n_hidden, epochs, batch, lr, decay, seed
    @staticmethod
    def _lr(x, a=0.01): return np.where(x >= 0, x, a * x)
    @staticmethod
    def _lrg(x, a=0.01): return np.where(x >= 0, 1.0, a)
    def _fwd(self, X):
        zs, hs = [], [X]
        for i, (W, b) in enumerate(self.P_):
            z = hs[-1] @ W + b; zs.append(z)
            hs.append(z if i == len(self.P_) - 1 else self._lr(z))
        return hs[-1].ravel(), zs, hs
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); rng = np.random.RandomState(self.seed)
        sizes = [X.shape[1]] + [self.width] * self.n_hidden + [1]
        self.P_ = [(rng.randn(a, b) * np.sqrt(2 / a), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]
        m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.P_]; v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.P_]
        t = 0; N = len(X); idx = np.arange(N)
        for ep in range(1, self.epochs + 1):
            rng.shuffle(idx); lr = self.lr * self.decay ** ep
            for s in range(0, N, self.batch):
                bi = idx[s:s + self.batch]; Xb, yb = X[bi], y[bi]
                yh, zs, hs = self._fwd(Xb)
                d = ((2 / len(bi)) * (yh - yb)).reshape(-1, 1); grads = []
                for i in reversed(range(len(self.P_))):
                    grads.append((hs[i].T @ d, d.sum(0)))
                    if i > 0: d = (d @ self.P_[i][0].T) * self._lrg(zs[i - 1])
                grads = grads[::-1]; t += 1; new = []
                for i, ((W, b), (gW, gb)) in enumerate(zip(self.P_, grads)):
                    mW, mb = m[i]; vW, vb = v[i]
                    mW = .9 * mW + .1 * gW; mb = .9 * mb + .1 * gb; vW = .999 * vW + .001 * gW ** 2; vb = .999 * vb + .001 * gb ** 2
                    m[i], v[i] = (mW, mb), (vW, vb)
                    new.append((W - lr * (mW / (1 - .9 ** t)) / (np.sqrt(vW / (1 - .999 ** t)) + 1e-8),
                                b - lr * (mb / (1 - .9 ** t)) / (np.sqrt(vb / (1 - .999 ** t)) + 1e-8)))
                self.P_ = new
        return self
    def predict(self, X): return self._fwd(np.asarray(X, float))[0]

class BondMeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.m_ = pd.Series(np.asarray(y)).groupby(X["Bond_Type"].values).mean().to_dict(); self.g_ = float(np.mean(y)); return self
    def predict(self, X): return np.array([self.m_.get(b, self.g_) for b in X["Bond_Type"].values])

def make_xgb(): return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=1, tree_method="hist", verbosity=0)
def make_rf():  return RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=SEED, n_jobs=-1)

MODELS = {
  "RF":  lambda: Pipeline([("sel", selector()), ("m", make_rf())]),
  "XGBoost": lambda: Pipeline([("sel", selector()), ("m", make_xgb())]),
  "DNN (descriptor-based)": lambda: Pipeline([("sel", selector()), ("sc", StandardScaler()),
                                              ("m", DescriptorDNN(epochs=DNN_EPOCHS, seed=SEED))]),
}
BASELINES = {
  "Bond-type mean": lambda: Pipeline([("sel", ColumnSelector(["Bond_Type"])), ("m", BondMeanRegressor())]),
  "Ridge (all curated inputs)": lambda: Pipeline([("sel", ColumnSelector(CURATED)), ("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
}
BOND_ID = ["Bond_Type", "bond_Ge_C", "bond_Ge_N", "bond_Ge_O"]; X_ID = ["EN_diff", "X_TotalValence", "X_degree"]
ABLATIONS = {
  "XGBoost, no explicit bond labels": lambda: Pipeline([("sel", ColumnSelector([c for c in CURATED if c not in BOND_ID])), ("m", make_xgb())]),
  "XGBoost, no bond labels, no X-identity descriptors": lambda: Pipeline([("sel", ColumnSelector([c for c in CURATED if c not in BOND_ID + X_ID])), ("m", make_xgb())]),
}
def metr(y, p):
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5,
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}

# %% [markdown]
# ## 4. Train, cross-validate (random and cluster-grouped) and test every model
# All preprocessing lives inside the pipeline, so each CV fold refits it on its own training part. The test set is touched once, at the end, by pipelines fitted on the 1,000 training compounds.

# %%
Xtr, Xte = tr[ALL_DESC], te[ALL_DESC]
ALLM = {**MODELS, **BASELINES, **ABLATIONS}
rows, PRED = [], pd.DataFrame({"compound_id": tr.compound_id}); PRED_TE = pd.DataFrame({"compound_id": te.compound_id, "broken_bond": te.broken_bond, "y_true": yte})
FITTED = {}
for name, f in ALLM.items():
    t0 = time.time()
    pr = cross_val_predict(f(), Xtr, ytr, cv=RANDOM_CV)
    pc = cross_val_predict(f(), Xtr, ytr, cv=CLUSTER_CV, groups=groups)
    fit = f().fit(Xtr, ytr); pt = fit.predict(Xte); FITTED[name] = fit
    PRED[name + " | cv_random"] = pr; PRED[name + " | cv_cluster"] = pc; PRED_TE[name] = pt
    for split, y_, p_ in [("CV random 5-fold (train)", ytr, pr), ("CV cluster 5-fold (train)", ytr, pc), ("External test (n=183)", yte, pt)]:
        rows.append({"Model": name, "Evaluation": split, **metr(y_, p_)})
    print(f"{name:55s} test R2={metr(yte, pt)['R2']:.4f}  ({time.time()-t0:.0f}s)")
RES = pd.DataFrame(rows).round(4); RES.to_csv(f"{OUT}/Table_model_performance.csv", index=False)
PRED.to_csv(f"{OUT}/predictions_cv_train.csv", index=False); PRED_TE.to_csv(f"{OUT}/predictions_external_test.csv", index=False)
print(RES.pivot(index="Model", columns="Evaluation", values="R2").to_string())
if FEATURE_MODE == "curated" or True:
    sel = FITTED["XGBoost"].named_steps["sel"]
    print("features used by XGBoost pipeline:", list(sel.transform(Xtr.head(1)).columns))

# %% [markdown]
# ## 5. Extra analyses requested by the reviewers

# %%
# 5a. Per-bond-class metrics on the external test set (and excluding the descriptor-duplicate test compounds)
pb = []
for name in list(MODELS) + list(BASELINES):
    for b in BOND_CODE:
        m = (te.broken_bond == b).values
        pb.append({"Model": name, "Bond": b, "n": int(m.sum()), **metr(yte[m], PRED_TE[name].values[m])})
    pb.append({"Model": name, "Bond": "all excl. 19 descriptor-duplicates", "n": int((~dup_mask).sum()), **metr(yte[~dup_mask], PRED_TE[name].values[~dup_mask])})
pd.DataFrame(pb).round(4).to_csv(f"{OUT}/Table_per_bond_class_test.csv", index=False)

# 5b. Conformal prediction intervals (Mondrian by bond class); calibration uses TRAINING cross-validated residuals only
def conformal(pred_train_cv, pred_test, level=0.9):
    res = np.abs(pred_train_cv - ytr); out = []; hw = {}
    for b in BOND_CODE:
        r = res[(tr.broken_bond == b).values]; k = min(1.0, np.ceil((len(r) + 1) * level) / len(r)); hw[b] = float(np.quantile(r, k))
    r = res; k = min(1.0, np.ceil((len(r) + 1) * level) / len(r)); hw["all"] = float(np.quantile(r, k))
    for b in list(BOND_CODE) + ["all"]:
        m = np.ones(len(te), bool) if b == "all" else (te.broken_bond == b).values
        out.append({"Bond": b, "n_test": int(m.sum()), "interval_half_width": round(hw[b], 3),
                    "coverage": round(float(np.mean(np.abs(pred_test[m] - yte[m]) <= hw[b])), 3), "nominal": level})
    return pd.DataFrame(out)
cp1 = conformal(PRED["XGBoost | cv_random"].values, PRED_TE["XGBoost"].values).assign(calibration="random-CV residuals")
cp2 = conformal(PRED["XGBoost | cv_cluster"].values, PRED_TE["XGBoost"].values).assign(calibration="cluster-CV residuals")
CP = pd.concat([cp1, cp2]); CP.to_csv(f"{OUT}/Table_conformal_intervals.csv", index=False); print(CP)

# 5c. SHAP for the XGBoost pipeline (fitted on training data only)
try:
    import shap
    xp = FITTED["XGBoost"]; Xs = xp.named_steps["sel"].transform(Xtr)
    sv = shap.TreeExplainer(xp.named_steps["m"]).shap_values(Xs)
    pd.DataFrame({"feature": Xs.columns, "mean_abs_SHAP": np.abs(sv).mean(0)}).sort_values("mean_abs_SHAP", ascending=False).to_csv(f"{OUT}/Table_SHAP_XGBoost.csv", index=False)
    plt.figure(); shap.summary_plot(sv, Xs, show=False); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_SHAP_XGBoost.png", dpi=300); plt.close()
except Exception as e: print("SHAP skipped:", e)

# 5d. Sensitivity: automatic filter refit inside folds (compare with the curated list)
if FEATURE_MODE == "curated":
    auto = Pipeline([("sel", selector("auto")), ("m", make_xgb())])
    pa = cross_val_predict(auto, Xtr, ytr, cv=RANDOM_CV); auto.fit(Xtr, ytr); pt_ = auto.predict(Xte)
    print("auto-filter (train-only, in-fold) XGBoost: CV", {k: round(v, 4) for k, v in metr(ytr, pa).items()},
          "| test", {k: round(v, 4) for k, v in metr(yte, pt_).items()}, "| features:", list(auto.named_steps["sel"].keep_))

# %% [markdown]
# ## 6. t-SNE of the Ge-X chemical space (exploratory, qualitative; coordinates are not interpretable axes)

# %%
from sklearn.manifold import TSNE
Zd = StandardScaler().fit_transform(D[CURATED].astype(float))
fp_all = np.array([list(gen.GetFingerprint(Chem.MolFromSmiles(s))) for s in D.SMILES], dtype=bool)
emb = {}
for tag, Z, kw in [("descriptors", Zd, {}), ("Morgan_fingerprint", fp_all, {"metric": "jaccard"})]:
    emb[tag] = TSNE(n_components=2, perplexity=30, init="random" if tag != "descriptors" else "pca", learning_rate="auto",
                    random_state=SEED, **kw).fit_transform(Z)
    pd.DataFrame({"compound_id": D.compound_id, "tsne1": emb[tag][:, 0], "tsne2": emb[tag][:, 1],
                  "broken_bond": D.broken_bond, "set": D.set, "BDE": D.y}).to_csv(f"{OUT}/tsne_{tag}.csv", index=False)
cols = {"Ge-C": "#2ca02c", "Ge-H": "#1f77b4", "Ge-N": "#ff7f0e", "Ge-O": "#d62728"}
fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
for r_, tag in enumerate(emb):
    E = emb[tag]
    for b, c in cols.items():
        m = (D.broken_bond == b).values; ax[r_, 0].scatter(E[m, 0], E[m, 1], s=9, c=c, label=b, alpha=.7)
    for s_, c, a in [("train", "#999999", .45), ("test", "#d4006e", .9)]:
        m = (D.set == s_).values; ax[r_, 1].scatter(E[m, 0], E[m, 1], s=9 if s_ == "train" else 14, c=c, label=s_, alpha=a)
    sc = ax[r_, 2].scatter(E[:, 0], E[:, 1], s=9, c=D.y, cmap="viridis"); plt.colorbar(sc, ax=ax[r_, 2], label="De (kcal/mol)")
    for k, t in enumerate(["coloured by broken bond", "training vs external test", "coloured by De"]):
        ax[r_, k].set_title(f"t-SNE of {tag.replace('_', ' ')}: {t}", fontsize=10); ax[r_, k].set_xticks([]); ax[r_, k].set_yticks([])
    ax[r_, 0].legend(markerscale=2, fontsize=8); ax[r_, 1].legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_tSNE.png", dpi=300); plt.close()

# %% [markdown]
# ## 7. Figures generated from stored predictions (nothing typed by hand)

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, name in zip(ax, MODELS):
    for b, c in cols.items():
        m = (te.broken_bond == b).values; a.scatter(yte[m], PRED_TE[name].values[m], s=14, c=c, label=b, alpha=.75)
    lim = [yte.min() - 2, yte.max() + 2]; a.plot(lim, lim, "k--", lw=1); mt = metr(yte, PRED_TE[name].values)
    a.set_title(f"{name}\nR2={mt['R2']:.3f}  MAE={mt['MAE']:.2f}  MAPE={mt['MAPE_%']:.2f}%", fontsize=10)
    a.set_xlabel("DFT De (kcal/mol)"); a.set_ylabel("Predicted De (kcal/mol)")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_parity_external_test.png", dpi=300); plt.close()
piv = RES[RES.Model.isin(list(MODELS) + list(BASELINES) + list(ABLATIONS))].pivot(index="Model", columns="Evaluation", values="MAE")
piv.plot.barh(figsize=(10, 5)); plt.xlabel("MAE (kcal/mol)"); plt.tight_layout(); plt.savefig(f"{OUT}/Fig_MAE_comparison.png", dpi=300); plt.close()

# %% [markdown]
# ## 8. Optional: TPOT (slow; run only if you must keep it)
# Search ONLY on the 1,000 training compounds, export the pipeline, then evaluate it once on the test set. The CV score of a pipeline chosen on the same data is optimistic: report it as such, or nest the search inside outer folds if time permits.

# %%
if RUN_TPOT:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tpot'])
    import tpot, joblib
    print("TPOT", tpot.__version__)
    Xc = tr[CURATED].astype(float).values
    tp = tpot.TPOTRegressor(generations=10, population_size=50, cv=5, random_state=SEED, n_jobs=1, max_time_mins=120, verbose=2)
    tp.fit(Xc, ytr)
    best = tp.fitted_pipeline_; print(best); joblib.dump(best, f"{OUT}/tpot_best_pipeline.pkl")
    open(f"{OUT}/tpot_best_pipeline.txt", "w").write(str(best))
    pt = best.predict(te[CURATED].astype(float).values); print("TPOT external test:", {k: round(v, 4) for k, v in metr(yte, pt).items()})
    pd.DataFrame({"compound_id": te.compound_id, "TPOT": pt}).to_csv(f"{OUT}/predictions_external_test_TPOT.csv", index=False)

# %% [markdown]
# ## 9. Manifest and download

# %%
json.dump({"seed": SEED, "feature_mode": FEATURE_MODE, "thresholds_auto": [TARGET_THR, PAIR_THR], "dnn_epochs": DNN_EPOCHS,
           "versions": VERS, "split": sim_summary, "n_dup_descriptor_test": int(dup_mask.sum()), "platform": platform.platform()},
          open(f"{OUT}/run_manifest.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))
try:
    import shutil; shutil.make_archive("outputs_Ge_X_BDE", "zip", OUT)
    from google.colab import files; files.download("outputs_Ge_X_BDE.zip")
except Exception: pass
