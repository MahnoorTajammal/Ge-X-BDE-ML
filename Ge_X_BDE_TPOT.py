"""TPOT (AutoML) for Ge-X De prediction. Search ONLY on the 1,000 training compounds; external set used once.
Inputs (from the main run's outputs folder, upload to Colab main folder): GE_BDE_descriptors.csv, split_assignments.csv
Run in Colab:  !pip -q install tpot xgboost scikit-learn   then   !python Ge_X_BDE_TPOT.py
Outputs go to tpot_outputs/ and tpot_outputs.zip. Needs ~1-2 h on free Colab (set QUICK=1 for a 3-minute smoke test)."""
import os, sys, json, time, random, warnings, shutil, importlib.metadata as md
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
SEED = 42; QUICK = bool(int(os.environ.get("QUICK", "0")))
GENERATIONS, POP, MAX_MIN = (2, 8, 5) if QUICK else (10, 50, 120)
OUT = "tpot_outputs"; os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
def find(name):
    for d in (".", "outputs"):
        if os.path.exists(os.path.join(d, name)): return os.path.join(d, name)
    raise FileNotFoundError(name + " not found: upload it to the Colab main folder")
D = pd.read_csv(find("GE_BDE_descriptors.csv")); S = pd.read_csv(find("split_assignments.csv"))
D = D.merge(S[["compound_id", "set"]], on="compound_id")
CURATED = ["MolWt","LogP","TPSA","HBD","HBA","EN_diff","X_TotalValence","X_degree","Ge_degree",
           "Ge_nb_C_count","Ge_nb_N_count","Ge_nb_O_count","Ge_nb_mean_EN","Ge_nb_max_EN",
           "Shell2_N_count","Shell2_O_count","bond_Ge_C","bond_Ge_N","bond_Ge_O","Bond_Type"]
tr = D[D.set == "train"].reset_index(drop=True); te = D[D.set == "test"].reset_index(drop=True)
assert len(tr) == 1000 and len(te) == 183
Xtr, Xte = tr[CURATED].astype(float).values, te[CURATED].astype(float).values; ytr, yte = tr.y.values, te.y.values
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def main():
    import tpot
    print("TPOT", tpot.__version__)
    tp = tpot.TPOTRegressor(search_space="linear", generations=GENERATIONS, population_size=POP, cv=5, n_jobs=1,
                            max_time_mins=MAX_MIN, max_eval_time_mins=10, random_state=SEED, verbose=2)
    t0 = time.time(); tp.fit(Xtr, ytr); mins = (time.time() - t0) / 60
    best = tp.fitted_pipeline_; print(best)
    joblib.dump(best, f"{OUT}/tpot_best_pipeline.pkl"); open(f"{OUT}/tpot_best_pipeline.txt", "w").write(str(best))
    def m(y, p): return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** .5, "MAE": mean_absolute_error(y, p), "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}
    pt = best.predict(Xte)
    # CV of the exported pipeline refit in each fold: OPTIMISTIC because the pipeline was chosen on these same 1,000 compounds
    pcv = cross_val_predict(clone(best), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=SEED))
    res = pd.DataFrame([{"Model": "TPOT", "Evaluation": "CV random 5-fold (train; optimistic, pipeline selected on same data)", **m(ytr, pcv)},
                        {"Model": "TPOT", "Evaluation": "External test (n=183)", **m(yte, pt)}]).round(4)
    res.to_csv(f"{OUT}/TPOT_results.csv", index=False); print(res.to_string())
    pd.DataFrame({"compound_id": te.compound_id, "broken_bond": te.broken_bond, "y_true": yte, "TPOT": pt}).to_csv(f"{OUT}/predictions_external_test_TPOT.csv", index=False)
    pd.DataFrame({"compound_id": tr.compound_id, "TPOT | cv_random": pcv}).to_csv(f"{OUT}/predictions_cv_train_TPOT.csv", index=False)
    vers = {p: md.version(p) for p in ["tpot", "scikit-learn", "numpy", "pandas"]}; vers["python"] = sys.version.split()[0]
    json.dump({"seed": SEED, "generations": GENERATIONS, "population_size": POP, "cv": 5, "max_time_mins": MAX_MIN,
               "search_minutes": round(mins, 1), "versions": vers}, open(f"{OUT}/tpot_manifest.json", "w"), indent=1)
    shutil.make_archive("tpot_outputs", "zip", OUT); print("done -> tpot_outputs.zip")


if __name__ == "__main__":
    main()
