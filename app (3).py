import ast
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Symptom Checker", layout="wide")

PRESETS = {
    "Chest pain": ["Pain chest", "Shortness of breath", "Sweating increased", "Nausea"],
    "Stomach trouble": ["Pain abdominal", "Vomiting", "Diarrhea", "Fever"],
    "Cough and fever": ["Cough", "Fever", "Chill", "Dyspnea"],
}
STEPS = ["Input", "Encode", "Weight", "Score", "Rank"]


def tidy(text):
    text = " ".join(text.replace("\xa0", " ").split())
    return (text[:1].upper() + text[1:]).replace("Hiv", "HIV")


@st.cache_data
def load_data():
    raw = pd.read_csv(Path(__file__).parent / "disease_data.csv")
    raw["Disease"] = raw["Disease"].map(tidy)
    raw["Symptoms"] = raw["Symptom"].map(lambda s: sorted({tidy(x) for x in ast.literal_eval(s) if x.strip()}))
    raw = raw.drop_duplicates("Disease").sort_values("Disease").reset_index(drop=True)
    symptoms = sorted({s for row in raw["Symptoms"] for s in row})
    matrix = pd.DataFrame(0, index=raw["Disease"], columns=symptoms)
    for name, row in zip(raw["Disease"], raw["Symptoms"]):
        matrix.loc[name, row] = 1
    return raw[["Disease", "Symptoms"]], matrix


def score(matrix, weights, chosen):
    picked = matrix[chosen]
    matched = picked.mul(weights[chosen]).sum(axis=1)
    chosen_w = weights[chosen].sum()
    total_w = matrix.mul(weights).sum(axis=1)
    out = pd.DataFrame(
        {
            "Disease": matrix.index,
            "Score": (matched / chosen_w).values,
            "Overlap": (matched / (chosen_w + total_w - matched)).values,
            "Matched": picked.sum(axis=1).values,
            "Total": matrix.sum(axis=1).values,
        }
    )
    out = out[out["Matched"] > 0].sort_values(["Score", "Overlap"], ascending=False)
    return out.reset_index(drop=True)


def flow_text(done):
    parts = []
    for i, name in enumerate(STEPS):
        colour = "green" if i < done else "orange" if i == done else "gray"
        parts.append(f":{colour}-background[{i + 1}. {name}]")
    return "  \u2192  ".join(parts)


def score_chart(top):
    base = alt.Chart(top).encode(y=alt.Y("Disease:N", sort="-x", title=None))
    bars = base.mark_bar(cornerRadiusEnd=5, height=22).encode(
        x=alt.X("Score:Q", axis=alt.Axis(format="%", title="Match score"), scale=alt.Scale(domain=[0, 1.1])),
        color=alt.Color("Score:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=["Disease", alt.Tooltip("Score:Q", format=".0%"), "Matched", "Total"],
    )
    labels = base.mark_text(align="left", dx=4).encode(x="Score:Q", text=alt.Text("Score:Q", format=".0%"))
    return (bars + labels).properties(height=34 * len(top), width="container")


def heatmap(top, matrix, chosen):
    rows = [
        {"Disease": d, "Symptom": s, "Listed": "Yes" if matrix.loc[d, s] else "No"}
        for d in top["Disease"]
        for s in chosen
    ]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect(stroke="white", strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X("Symptom:N", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("Disease:N", sort=list(top["Disease"]), title=None),
            color=alt.Color(
                "Listed:N",
                scale=alt.Scale(domain=["Yes", "No"], range=["#2E9E6B", "#D5DBE3"]),
                legend=alt.Legend(title="Disease lists symptom"),
            ),
            tooltip=["Disease", "Symptom", "Listed"],
        )
        .properties(height=34 * len(top), width="container")
    )


def weight_chart(chosen, weights):
    data = pd.DataFrame({"Symptom": chosen, "Weight": weights[chosen].values})
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=5, color="#E8A33D", height=20)
        .encode(
            x=alt.X("Weight:Q", title="Rarity weight (higher = more telling)"),
            y=alt.Y("Symptom:N", sort="-x", title=None),
            tooltip=["Symptom", alt.Tooltip("Weight:Q", format=".2f")],
        )
        .properties(height=32 * len(chosen), width="container")
    )


def run_pipeline(chosen, matrix, weights):
    flow = st.empty()
    flow.markdown(flow_text(0))
    with st.status("Running prediction pipeline", expanded=True) as status:
        flow.markdown(flow_text(0))
        st.markdown("**1. Input** \u2013 symptoms received")
        st.markdown(" ".join(f":blue-background[{s}]" for s in chosen))
        time.sleep(0.8)

        flow.markdown(flow_text(1))
        st.markdown("**2. Encode** \u2013 symptoms turned into a yes/no vector")
        st.write(f"The vector has {matrix.shape[1]} symptom slots, and {len(chosen)} of them are switched on.")
        time.sleep(0.8)

        flow.markdown(flow_text(2))
        st.markdown("**3. Weight** \u2013 rare symptoms count for more than common ones")
        st.altair_chart(weight_chart(chosen, weights))
        time.sleep(1.0)

        flow.markdown(flow_text(3))
        st.markdown("**4. Score** \u2013 comparing against every disease")
        bar = st.progress(0.0)
        total = len(matrix)
        for i in range(1, 11):
            time.sleep(0.12)
            bar.progress(i / 10, text=f"Compared {round(total * i / 10)} of {total} diseases")
        ranked = score(matrix, weights, chosen)
        time.sleep(0.4)

        flow.markdown(flow_text(4))
        st.markdown("**5. Rank** \u2013 sorting by match score")
        if ranked.empty:
            st.warning("No disease lists these symptoms.")
        else:
            best = ranked.iloc[0]
            st.write(f"Best match: **{best['Disease']}** ({best['Score']:.0%})")
        time.sleep(0.6)

        flow.markdown(flow_text(5))
        status.update(label="Pipeline complete", state="complete", expanded=False)
    st.session_state["result"] = {"chosen": chosen, "ranked": ranked}


def show_results(result, matrix):
    chosen, ranked = result["chosen"], result["ranked"]
    if ranked.empty:
        return
    top = ranked.head(8)
    best = top.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top match", best["Disease"])
    c2.metric("Match score", f"{best['Score']:.0%}")
    c3.metric("Symptoms matched", f"{int(best['Matched'])} of {len(chosen)}")
    left, right = st.columns(2)
    with left:
        st.subheader("Top matches")
        st.altair_chart(score_chart(top))
    with right:
        st.subheader("Why these matches")
        st.altair_chart(heatmap(top, matrix, chosen))
    st.caption("Educational demo on a small dataset. Not medical advice.")


def apply_preset(name):
    st.session_state["chosen"] = PRESETS[name]


_, matrix = load_data()
doc_freq = matrix.sum(axis=0)
weights = np.log((len(matrix) + 1) / (doc_freq + 1)) + 1

st.title("Symptom Checker")

st.multiselect("Select your symptoms (type to search)", matrix.columns.tolist(), key="chosen")
cols = st.columns(len(PRESETS) + 1)
cols[0].write("Quick examples:")
for col, label in zip(cols[1:], PRESETS):
    col.button(label, on_click=apply_preset, args=(label,))
chosen = st.session_state.get("chosen", [])
if st.button("Predict", type="primary", disabled=not chosen):
    run_pipeline(list(chosen), matrix, weights)
if "result" in st.session_state:
    show_results(st.session_state["result"], matrix)
