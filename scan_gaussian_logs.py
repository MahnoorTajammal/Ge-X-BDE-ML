"""Scan Gaussian 16 Opt+Freq logs for a Ge-X dataset (folders, .log files or .zip archives).
Usage:  python scan_gaussian_logs.py LOG_OR_FOLDER_OR_ZIP [more ...] [--dataset GE_BDE.csv] [--out scan_results]
Needs: pandas, openpyxl, numpy.
Per species (parent/frag1/frag2) it reports: termination status, error text, multiplicity, <S2> before/after annihilation,
final SCF energy, lowest frequency / number of imaginary frequencies, ZPE and enthalpy. Per compound it reports De (= E_frag1+E_frag2-E_parent),
D0 and DH298 and, with --dataset, the difference to the tabulated value.
It also writes (1) restart .gjf files for jobs that failed, starting from their last geometry and (2) re-optimisation .gjf files for
species with an imaginary frequency, displaced along that mode. Run the .gjf files, put the new logs in the scan and run again."""
import re, sys, os, zipfile, argparse, numpy as np, pandas as pd
H = 627.5095
SYM = "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split()
ERR = re.compile(r"(Error termination[^\n]*|Convergence failure[^\n]*|FormBX had a problem|Error in internal coordinate system|Operation on file out of range|Segmentation|galloc|Erroneous write|Bend failed[^\n]*|Linear angle[^\n]*)")
def read_texts(paths):
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for f in sorted(fs):
                    if f.lower().endswith(".log"): yield f, open(os.path.join(r, f), errors="ignore").read()
        elif p.lower().endswith(".zip"):
            z = zipfile.ZipFile(p)
            for n in sorted(z.namelist()):
                if n.lower().endswith(".log"): yield n, z.read(n).decode(errors="ignore")
        else: yield os.path.basename(p), open(p, errors="ignore").read()
def last_geom(t):
    blocks = re.findall(r"Input orientation:.*?\n -+\n.*?\n -+\n(.*?)\n -+\n", t, re.S)
    if not blocks: return None
    out = []
    for l in blocks[-1].split("\n"):
        x = l.split()
        if len(x) == 6: out.append((SYM[int(x[1])], float(x[3]), float(x[4]), float(x[5])))
    return out
def imag_mode(t, geom):
    for m in re.finditer(r"Frequencies --\s+(.*)\n", t):
        fq = [float(v) for v in m.group(1).split()]
        if fq and fq[0] < 0 or any(v < 0 for v in fq):
            k = [i for i, v in enumerate(fq) if v < 0][0]
            tail = t[m.end():m.end() + 400000].split("\n")
            i = next((j for j, l in enumerate(tail) if l.strip().startswith("Atom") and "AN" in l), None)
            if i is None: return None
            vec = []
            for l in tail[i + 1:i + 1 + len(geom)]:
                x = l.split(); vec.append([float(v) for v in x[2 + 3 * k:5 + 3 * k]])
            return fq[k], np.array(vec)
    return None
def parse_job(seg_list):
    t = "\n".join(seg_list)
    title = re.search(r"\n (compound_\w+?_(?:parent|frag1|frag2))\s*\n", t)
    route = re.search(r"\n (#[^\n]*)", t); mult = re.search(r"Charge =\s*(-?\d+) Multiplicity = (\d+)", t)
    scf = re.findall(r"SCF Done:\s+E\((\S+)\)\s+=\s+(-?[\d.]+)", t); fq = [float(v) for l in re.findall(r"Frequencies --\s+(.*)", t) for v in l.split()]
    s2 = re.findall(r"S\*\*2 before annihilation\s+([\d.]+),\s+after\s+([\d.]+)", t); g = lambda p: float(re.search(p, t).group(1)) if re.search(p, t) else np.nan
    nt = len(re.findall("Normal termination", t)); er = ERR.search(t)
    nat = re.search(r"NAtoms=\s+(\d+)", t); single = bool(nat) and int(nat.group(1)) == 1
    status = "OK" if (nt >= 1 and (fq != [] or single) and not er) else ("FAILED" if er or nt == 0 else "INCOMPLETE")
    return t, dict(job=title.group(1) if title else None, route=route.group(1)[:80] if route else None, charge=int(mult.group(1)) if mult else None, mult=int(mult.group(2)) if mult else None,
        status=status, normal_terminations=nt, error=er.group(1)[:90] if er else "", scf_label=scf[-1][0] if scf else None, E=float(scf[-1][1]) if scf else np.nan,
        S2_before=float(s2[-1][0]) if s2 else np.nan, S2_after=float(s2[-1][1]) if s2 else np.nan, n_freq=len(fq), n_imag=sum(v < 0 for v in fq), lowest_freq=min(fq) if fq else np.nan,
        ZPE=g(r"Zero-point correction=\s+(-?[\d.]+)"), H298=g(r"Sum of electronic and thermal Enthalpies=\s+(-?[\d.]+)"))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--dataset"); ap.add_argument("--out", default="scan_results"); a = ap.parse_args()
    os.makedirs(a.out + "/restart_gjf", exist_ok=True); os.makedirs(a.out + "/reopt_imag_gjf", exist_ok=True); rows = []; seen = {}
    for name, txt in read_texts(a.inputs):
        segs = re.split(r"\n Entering Link 1", txt); cur = []
        groups = []
        for s in segs[1:]:
            if re.search(r"\n compound_\w+?_(?:parent|frag1|frag2)\s*\n", s) and cur: groups.append(cur); cur = [s]
            else: cur.append(s)
        if cur: groups.append(cur)
        for gs in groups:
            t, r = parse_job(gs); r["file"] = name
            if not r["job"]: continue
            geom = last_geom(t); rad = (r["mult"] or 1) > 1; method = ("UCAM-B3LYP" if rad else "CAM-B3LYP") + "/LANL2DZ"
            if r["status"] != "OK" and geom:
                with open(f"{a.out}/restart_gjf/{r['job']}.gjf", "w") as f:
                    f.write(f"%nprocshared=4\n%mem=8GB\n#p scf=(converge=12,maxcycle=512) {method} Opt=(CalcFC,MaxCycles=300) Freq\n\n{r['job']}\n\n{r['charge']} {r['mult']}\n" + "\n".join(f"{s:<2s} {x:12.6f} {y:12.6f} {z:12.6f}" for s, x, y, z in geom) + "\n\n")
            if r["status"] == "OK" and r["n_imag"] > 0 and geom:
                im = imag_mode(t, geom)
                if im:
                    f0, v = im; v = v / max(np.linalg.norm(v, axis=1).max(), 1e-6) * 0.4
                    with open(f"{a.out}/reopt_imag_gjf/{r['job']}.gjf", "w") as f:
                        f.write(f"%nprocshared=4\n%mem=8GB\n#p scf=(converge=12,maxcycle=512) {method} Opt=(Tight,CalcFC,MaxCycles=300) Freq\n\n{r['job']}_displaced\n\n{r['charge']} {r['mult']}\n" + "\n".join(f"{s:<2s} {x+d[0]:12.6f} {y+d[1]:12.6f} {z+d[2]:12.6f}" for (s, x, y, z), d in zip(geom, v)) + "\n\n")
                    r["imag_mode_cm1"] = f0
            seen[r["job"]] = r   # later logs (restarts) overwrite earlier failed ones
    for jn, r in seen.items():   # drop stale restart/re-opt inputs for jobs that a later log fixed
        if r["status"] == "OK" and os.path.exists(f"{a.out}/restart_gjf/{jn}.gjf"): os.remove(f"{a.out}/restart_gjf/{jn}.gjf")
        if r["n_imag"] == 0 and os.path.exists(f"{a.out}/reopt_imag_gjf/{jn}.gjf"): os.remove(f"{a.out}/reopt_imag_gjf/{jn}.gjf")
    J = pd.DataFrame(seen.values()); J["compound"] = J.job.str.rsplit("_", n=1).str[0]; J["species"] = J.job.str.rsplit("_", n=1).str[1]
    comp = []
    for c, g in J.groupby("compound"):
        d = {s: g[g.species == s].iloc[0] for s in ["parent", "frag1", "frag2"] if (g.species == s).any()}
        r = dict(compound=c, status="complete" if len(d) == 3 and all(v.status == "OK" for v in d.values()) else "needs rerun")
        if r["status"] == "complete":
            p, f1, f2 = d["parent"], d["frag1"], d["frag2"]
            r.update(De=(f1.E + f2.E - p.E) * H, D0=(f1.E + f1.ZPE + f2.E + f2.ZPE - p.E - p.ZPE) * H, DH298=(f1.H298 + f2.H298 - p.H298) * H, max_S2_before=np.nanmax([f1.S2_before, f2.S2_before]),
                     n_imag_species=int((np.array([p.n_imag, f1.n_imag, f2.n_imag]) > 0).sum()))
        comp.append(r)
    C = pd.DataFrame(comp)
    if a.dataset:
        D = pd.read_csv(a.dataset, encoding="utf-8-sig"); D = D.rename(columns={D.columns[0]: "compound", D.columns[-1]: "dataset_BDE"})[["compound", "dataset_BDE"]]
        C = C.merge(D, on="compound", how="left"); C["De - dataset"] = C.De - C.dataset_BDE
    with pd.ExcelWriter(a.out + "/scan_summary.xlsx", engine="openpyxl") as w:
        J.to_excel(w, sheet_name="jobs", index=False); C.to_excel(w, sheet_name="compounds", index=False)   # full precision: do not round energies
        sp = J[(J.mult > 1) & J.S2_before.notna()]
        pd.DataFrame([dict(n_unrestricted_jobs=len(sp), S2_before_min=sp.S2_before.min(), S2_before_mean=sp.S2_before.mean(), S2_before_max=sp.S2_before.max(), S2_after_min=sp.S2_after.min(), S2_after_max=sp.S2_after.max(),
             n_jobs=len(J), n_ok=int((J.status == "OK").sum()), n_failed_or_incomplete=int((J.status != "OK").sum()), n_species_with_imag_freq=int((J.n_imag > 0).sum()))]).round(4).to_excel(w, sheet_name="summary", index=False)
        J[J.status != "OK"][["job", "file", "status", "error", "normal_terminations"]].to_excel(w, sheet_name="needs_restart", index=False)
        J[J.n_imag > 0][["job", "n_imag", "lowest_freq", "file"]].to_excel(w, sheet_name="imaginary_freq", index=False)
    print(pd.read_excel(a.out + "/scan_summary.xlsx", sheet_name="summary").T.to_string(header=False)); print("wrote", a.out + "/scan_summary.xlsx")
if __name__ == "__main__": main()
