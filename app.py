import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Secure Federated Healthcare AI", page_icon="🏥", layout="wide")
st.title("🏥 Secure Federated Learning for Healthcare")
st.caption("Hospitals train one shared screening model together. Patient data never leaves the hospital. Uses simulated patient records, no real patient data.")

COND = ["Diabetes", "Heart disease", "Hypertension", "Kidney disease"]
HOSP = [("General Hospital", None), ("Heart Institute", [1, 2]), ("Diabetes Centre", [0]), ("Kidney Care", [3]), ("Geriatric Hospital", "old"), ("Community Clinic", None)]
STEPS = ["Broadcast", "Local training", "Clip + noise", "Secure masking", "FedAvg", "Ledger block"]
FIELDS = ["Age", "Sex", "BMI", "Systolic BP", "Fasting glucose", "Total cholesterol", "HDL", "Creatinine", "Smoker", "Heart rate"]
PRESETS = {
    "Healthy adult": [35, "Female", 23.0, 115, 88, 175, 58, 0.8, False, 68],
    "Overweight, high sugar": [48, "Male", 32.0, 132, 165, 215, 40, 0.95, False, 78],
    "Hypertensive smoker": [62, "Male", 29.0, 162, 105, 245, 38, 1.0, True, 84],
    "Elderly, kidney concerns": [74, "Female", 27.0, 148, 120, 205, 48, 2.4, False, 76],
}

with st.sidebar:
    k = st.slider("Hospitals", 3, 6, 3)
    rounds = st.slider("Rounds", 5, 30, 12)
    dp = st.toggle("Differential privacy", True)
    sigma = st.slider("Noise level", 0.05, 2.0, 0.1, 0.05)
    secagg = st.toggle("Secure aggregation", True)
    delay = st.slider("Animation delay (s)", 0.0, 0.5, 0.15, 0.05)
    go = st.button("Start training", type="primary")


def sigm(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def population(n, rng):
    age = np.clip(rng.normal(52, 16, n), 18, 90)
    sex = (rng.random(n) < 0.5).astype(float)
    bmi = np.clip(rng.normal(26 + 0.03 * (age - 50), 4.5), 15, 50)
    smk = (rng.random(n) < 0.2).astype(float)
    dia = rng.random(n) < sigm(-2.6 + 0.03 * (age - 50) + 0.1 * (bmi - 26))
    htn = rng.random(n) < sigm(-1.5 + 0.04 * (age - 50) + 0.06 * (bmi - 26))
    hrt = rng.random(n) < sigm(-3.3 + 0.045 * (age - 50) + 0.8 * smk + 0.5 * sex + 0.5 * htn + 0.5 * dia)
    ckd = rng.random(n) < sigm(-3.6 + 0.04 * (age - 50) + 0.8 * dia + 0.8 * htn)
    glu = np.clip(rng.normal(92 + 0.1 * (age - 50) + 0.8 * (bmi - 26), 9) + dia * np.abs(rng.normal(60, 30, n)), 60, 360)
    sbp = np.clip(rng.normal(112 + 0.3 * (age - 50) + 0.5 * (bmi - 26), 9) + htn * np.abs(rng.normal(26, 10, n)), 85, 220)
    chol = np.clip(rng.normal(180 + 0.3 * (age - 50), 28) + hrt * np.abs(rng.normal(30, 20, n)) + dia * 10, 100, 340)
    hdl = np.clip(rng.normal(55 - 6 * sex - 0.5 * (bmi - 26), 9) - hrt * 6, 20, 100)
    cre = np.clip(rng.normal(0.82 + 0.22 * sex + 0.003 * (age - 50), 0.12) + ckd * np.abs(rng.normal(0.9, 0.5, n)), 0.4, 5)
    hr = np.clip(rng.normal(70 + 3 * smk + 5 * hrt, 8), 45, 130)
    return np.column_stack([age, sex, bmi, sbp, glu, chol, hdl, cre, smk, hr]), np.column_stack([dia, hrt, htn, ckd]).astype(float)


@st.cache_data
def load(k):
    ref, _ = population(20000, np.random.default_rng(99))
    mu, sd = ref.mean(0), ref.std(0)
    Xp, Yp = population(700, np.random.default_rng(5))
    Xt, Yt = population(1500, np.random.default_rng(6))
    w = np.ones((len(Xp), k))
    for h in range(k):
        tgt = HOSP[h][1]
        if tgt == "old":
            w[:, h] += 8 * (Xp[:, 0] > 65)
        elif tgt:
            w[:, h] += 25 * Yp[:, tgt].max(1)
    p = w / w.sum(1, keepdims=True)
    u = np.random.default_rng(1).random(len(Xp))
    a = np.minimum((u[:, None] > np.cumsum(p, 1)).sum(1), k - 1)
    return Xp, Yp, a, Xt, Yt, mu, sd


def feats(X, mu, sd):
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))])


def train(W, X, Y, epochs=2, lr=0.1):
    W = W.copy()
    for _ in range(epochs):
        for i in range(0, len(X), 32):
            xb, yb = X[i:i + 32], Y[i:i + 32]
            W -= lr * xb.T @ (sigm(xb @ W) - yb) / len(xb)
    return W


def auc(s, y):
    if y.min() == y.max():
        return 0.5
    r = s.argsort().argsort() + 1
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def aucs(W, Z, Y):
    S = Z @ W
    return [auc(S[:, j], Y[:, j]) for j in range(Y.shape[1])]


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@st.fragment
def predict_ui():
    m = st.session_state.get("m")
    if not m:
        st.info("Train the model first, then come back here to screen a patient.")
        return
    preset = st.selectbox("Patient profile", list(PRESETS) + ["Custom"])
    base = PRESETS.get(preset, PRESETS["Healthy adult"])
    c = st.columns(3)
    v = [
        c[0].slider("Age", 18, 90, base[0], key=f"a{preset}"),
        c[1].radio("Sex", ["Female", "Male"], index=int(base[1] == "Male"), horizontal=True, key=f"s{preset}"),
        c[2].slider("BMI", 15.0, 50.0, base[2], 0.5, key=f"b{preset}"),
        c[0].slider("Systolic BP (mmHg)", 85, 220, base[3], key=f"p{preset}"),
        c[1].slider("Fasting glucose (mg/dL)", 60, 360, base[4], key=f"g{preset}"),
        c[2].slider("Total cholesterol (mg/dL)", 100, 340, base[5], key=f"c{preset}"),
        c[0].slider("HDL (mg/dL)", 20, 100, base[6], key=f"h{preset}"),
        c[1].slider("Creatinine (mg/dL)", 0.4, 5.0, base[7], 0.1, key=f"r{preset}"),
        c[2].toggle("Smoker", base[8], key=f"k{preset}"),
        c[0].slider("Heart rate (bpm)", 45, 130, base[9], key=f"hr{preset}"),
    ]
    x = np.array([[v[0], float(v[1] == "Male"), v[2], v[3], v[4], v[5], v[6], v[7], float(v[8]), v[9]]])
    z = feats(x, m["mu"], m["sd"])
    risk = sigm(z @ m["W"])[0]
    st.markdown("### Screening result (federated model)")
    cols = st.columns(4)
    for j, name in enumerate(COND):
        level = "🟢 Low" if risk[j] < 0.2 else "🟡 Moderate" if risk[j] < 0.5 else "🔴 High"
        cols[j].metric(name, f"{risk[j] * 100:.0f}%", level, delta_color="off")
        cols[j].progress(float(risk[j]))
    top = int(risk.argmax())
    if risk[top] < 0.2:
        st.success("No elevated screening risk detected for the four conditions.")
    else:
        st.warning(f"Highest risk: **{COND[top]}** ({risk[top] * 100:.0f}%). Recommend clinical follow-up and confirmatory tests.")
    rows = {"Federated": risk, "Centralized": sigm(z @ m["Wc"])[0]}
    for n, Wl in enumerate(m["Wl"]):
        rows[f"{HOSP[n][0]} only"] = sigm(z @ Wl)[0]
    st.markdown("**Same patient, different models (risk %)**")
    st.dataframe(pd.DataFrame({n: [f"{x_ * 100:.0f}%" for x_ in r] for n, r in rows.items()}, index=COND))
    st.caption("Educational demo on simulated data. Not medical advice.")


tab_train, tab_pred = st.tabs(["🔬 Training process", "🩺 Screen a patient"])

with tab_train:
    if go:
        rng = np.random.default_rng(1)
        Xp, Yp, a, Xt, Yt, mu, sd = load(k)
        Zp, Zt = feats(Xp, mu, sd), feats(Xt, mu, sd)
        hospitals = [(Zp[a == h], Yp[a == h]) for h in range(k)]
        sizes = np.array([len(h[1]) for h in hospitals], dtype=float)
        weights = sizes / sizes.sum()
        W = np.zeros((Zp.shape[1], len(COND)))
        chain = [hashlib.sha256(b"genesis").hexdigest()]
        curve, log = [], []

        st.markdown("**What each hospital sees** (% of its patients with each condition)")
        st.dataframe(pd.DataFrame(
            [[HOSP[h][0], int(sizes[h])] + [f"{Yp[a == h][:, j].mean() * 100:.0f}%" for j in range(4)] for h in range(k)],
            columns=["Hospital", "Patients"] + COND,
        ), hide_index=True)

        left, right = st.columns([3, 2])
        flow = left.empty()
        status = left.empty()
        ledger = left.empty()
        chart = right.empty()
        console = right.empty()
        bar = st.progress(0.0)
        names = "  ".join(f"🏥 {HOSP[h][0]} ({int(sizes[h])})" for h in range(k))

        def show(step, r, msg):
            parts = [f"✅ {s}" if i < step else f"▶️ **{s}**" if i == step else f"⬜ {s}" for i, s in enumerate(STEPS)]
            flow.markdown(f"### Round {r} of {rounds}\n" + "  →  ".join(parts))
            status.info(f"{names}\n\n🔒 Raw patient records stay inside each hospital")
            log.append(f"[R{r:02d}] {msg}")
            console.code("\n".join(log[-9:]))
            ledger.code("⛓ Ledger\n" + "\n".join(f"#{len(chain) - 3 + i}  {h[:24]}…" for i, h in enumerate(chain[-3:])))
            time.sleep(delay)

        for r in range(1, rounds + 1):
            show(0, r, "server sends global model to hospitals")
            deltas = [train(W, X, Y) - W for X, Y in hospitals]
            show(1, r, "each hospital trained on its own patients")
            if dp:
                clipped = [d * min(1.0, 1.0 / (np.linalg.norm(d) + 1e-12)) for d in deltas]
                std = sigma * weights.max() / np.sqrt(k)
                ups = [weights[i] * clipped[i] + rng.normal(0, std, size=W.shape) for i in range(k)]
                show(2, r, f"updates clipped and noised (sigma={sigma})")
            else:
                ups = [weights[i] * deltas[i] for i in range(k)]
                show(2, r, "differential privacy off")
            if secagg:
                masked = [u.copy() for u in ups]
                for i in range(k):
                    for j in range(i + 1, k):
                        mk = np.random.default_rng(r * 1000 + i * 10 + j).normal(0, 5, size=W.shape)
                        masked[i] += mk
                        masked[j] -= mk
                show(3, r, f"masked updates sent, server sees noise (similarity {cos(masked[0], ups[0]):+.2f})")
            else:
                masked = ups
                show(3, r, "secure aggregation off, server sees raw updates")
            W = W + np.sum(masked, axis=0)
            curve.append(float(np.mean(aucs(W, Zt, Yt))))
            show(4, r, f"FedAvg done, mean AUC {curve[-1]:.3f}")
            chain.append(hashlib.sha256((chain[-1] + W.tobytes().hex()).encode()).hexdigest())
            chart.line_chart({"Mean AUC across 4 conditions": curve})
            show(5, r, f"block #{len(chain) - 1} added {chain[-1][:12]}…")
            bar.progress(r / rounds)

        Wc = train(np.zeros_like(W), Zp, Yp, epochs=rounds * 2)
        Wl = [train(np.zeros_like(W), X, Y, epochs=rounds * 2) for X, Y in hospitals]
        st.session_state["m"] = {"W": W, "Wc": Wc, "Wl": Wl, "mu": mu, "sd": sd}
        fa, ca = aucs(W, Zt, Yt), aucs(Wc, Zt, Yt)
        la = np.mean([aucs(x, Zt, Yt) for x in Wl], axis=0)
        st.success("Training complete. Open the Screen a patient tab to use the federated model.")
        x1, x2, x3 = st.columns(3)
        x1.metric("Federated (mean AUC)", f"{np.mean(fa):.3f}")
        x2.metric("Centralized (needs pooled data)", f"{np.mean(ca):.3f}")
        x3.metric("Hospital-only average", f"{np.mean(la):.3f}")
        st.dataframe(pd.DataFrame({"Federated": fa, "Centralized": ca, "Hospital-only avg": la}, index=COND).round(3))
    else:
        st.info("Set options in the sidebar and press Start training.")

with tab_pred:
    predict_ui()
