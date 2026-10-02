import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Symptom Checker Demo", layout="wide")

PRESETS = {
    "Chest pain": ["pain chest", "shortness of breath", "sweating increased", "nausea"],
    "Stomach trouble": ["pain abdominal", "vomiting", "diarrhea", "fever"],
    "Cough and fever": ["cough", "fever", "chill", "dyspnea"],
}
MODES = ["Explains my symptoms best", "Closest overall profile"]


def tidy(text):
    return " ".join(text.replace("\xa0", " ").split())


@st.cache_data
def load_data():
    raw = pd.read_csv(Path(__file__).parent / "disease_data.csv")
    raw["Disease"] = raw["Disease"].map(tidy)
    raw["Symptoms"] = raw["Symptom"].map(lambda s: sorted({tidy(x) for x in ast.literal_eval(s)}))
    raw = raw.drop_duplicates("Disease").sort_values("Disease").reset_index(drop=True)
    symptoms = sorted({s for row in raw["Symptoms"] for s in row})
    matrix = pd.DataFrame(0, index=raw["Disease"], columns=symptoms)
    for name, row in zip(raw["Disease"], raw["Symptoms"]):
        matrix.loc[name, row] = 1
    return raw[["Disease", "Symptoms"]], matrix


def rank(matrix, weights, chosen, mode):
    picked = matrix[chosen]
    matched_w = picked.mul(weights[chosen]).sum(axis=1)
    chosen_w = weights[chosen].sum()
    disease_w = matrix.mul(weights).sum(axis=1)
    coverage = matched_w / chosen_w
    overlap = matched_w / (chosen_w + disease_w - matched_w)
    out = pd.DataFrame(
        {
            "Disease": matrix.index,
            "Coverage": coverage.values,
            "Overlap": overlap.values,
            "Matched": picked.sum(axis=1).values,
            "Total": matrix.sum(axis=1).values,
        }
    )
    main, second = ("Coverage", "Overlap") if mode == MODES[0] else ("Overlap", "Coverage")
    out["Score"] = out[main]
    out = out.sort_values(["Score", second, "Matched"], ascending=False)
    return out[out["Matched"] > 0].reset_index(drop=True)


def apply_preset(name):
    st.session_state["chosen"] = PRESETS[name]


diseases, matrix = load_data()
doc_freq = matrix.sum(axis=0)
weights = np.log((len(matrix) + 1) / (doc_freq + 1)) + 1

st.title("Symptom Checker Demo")
st.caption("Educational demo built on a small disease-symptom dataset. Not medical advice.")

with st.sidebar:
    st.header("Your symptoms")
    st.multiselect("Pick symptoms (type to search)", matrix.columns.tolist(), key="chosen")
    st.write("Quick examples")
    for label in PRESETS:
        st.button(label, on_click=apply_preset, args=(label,))
    mode = st.radio("Ranking style", MODES)
    top_n = st.slider("Diseases to show", 3, 20, 8)

chosen = st.session_state.get("chosen", [])
tab_results, tab_explore, tab_about = st.tabs(["Results", "Explore data", "How it works"])

with tab_results:
    if not chosen:
        st.info("Choose one or more symptoms in the sidebar, or click a quick example.")
    else:
        ranked = rank(matrix, weights, chosen, mode)
        if ranked.empty:
            st.warning("No disease in the dataset lists those symptoms.")
        else:
            top = ranked.head(top_n)
            best = top.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Top match", best["Disease"])
            c2.metric("Match score", f"{best['Score']:.0%}")
            c3.metric("Symptoms matched", f"{int(best['Matched'])} of {len(chosen)} chosen")

            st.bar_chart(top.set_index("Disease")["Score"])
            st.dataframe(
                top[["Disease", "Score", "Matched", "Total"]],
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
                    "Matched": "Chosen symptoms matched",
                    "Total": "Symptoms listed for disease",
                },
                hide_index=True,
            )

            st.subheader("Why these matches")
            lookup = diseases.set_index("Disease")["Symptoms"]
            for name in top["Disease"].head(3):
                with st.expander(name, expanded=False):
                    hit = [s for s in lookup[name] if s in chosen]
                    rest = [s for s in lookup[name] if s not in chosen]
                    st.markdown("**Matched:** " + ", ".join(hit))
                    st.markdown("**Other symptoms listed:** " + (", ".join(rest) or "none"))

            leads = matrix.loc[top["Disease"].head(5)].drop(columns=chosen).sum().sort_values(ascending=False)
            leads = leads[leads > 0].head(8)
            if not leads.empty:
                st.subheader("Worth checking next")
                st.write("Symptoms common among the top matches that you have not selected: " + ", ".join(leads.index))

with tab_explore:
    c1, c2, c3 = st.columns(3)
    c1.metric("Diseases", len(diseases))
    c2.metric("Unique symptoms", matrix.shape[1])
    c3.metric("Avg symptoms per disease", f"{matrix.sum(axis=1).mean():.1f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Browse a disease")
        pick = st.selectbox("Disease", diseases["Disease"])
        st.write(", ".join(diseases.set_index("Disease").loc[pick, "Symptoms"]))
    with right:
        st.subheader("Most common symptoms")
        st.bar_chart(doc_freq.sort_values(ascending=False).head(15))

with tab_about:
    st.markdown(
        """
        Each disease in the dataset has a list of symptoms. There is one row per disease, so this demo
        ranks diseases by symptom overlap rather than training a model.

        **Weighting:** rare symptoms count for more than common ones such as pain or fever (inverse document frequency).

        **Ranking styles**
        - *Explains my symptoms best*: the share of your weighted symptoms that a disease covers.
        - *Closest overall profile*: weighted Jaccard overlap, which also penalises diseases with many unrelated symptoms.

        This is a teaching demo. It cannot diagnose anything. See a clinician for real health concerns.
        """
    )
