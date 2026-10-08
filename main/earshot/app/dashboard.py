import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Earshot", layout="wide")
st.title("Earshot")

live_tab, replay_tab = st.tabs(["Live fan", "Rack replay"])

with live_tab:
    log_path = st.text_input("live log", "runs/live/scores.csv")

    @st.fragment(run_every=1.0)
    def live_panel():
        path = Path(log_path)
        if not path.exists():
            st.info("start `python -m earshot.audio.live <bundle>` to see scores here")
            return
        df = pd.read_csv(path).tail(240)
        if df.empty:
            return
        df["seconds"] = df["time"] - df["time"].iloc[0]
        health = df["health"].iloc[-1]
        col1, col2 = st.columns([1, 3])
        col1.metric("fan health", f"{health:.2f}", delta="check fan" if health < 0.5 else "ok",
                    delta_color="inverse" if health < 0.5 else "normal")
        col2.line_chart(df.set_index("seconds")[["health"]], height=260)

    live_panel()

with replay_tab:
    traces = sorted(Path("results").glob("trace_*.npz"))
    if not traces:
        st.info("run `python -m earshot.control.evaluate ...` first")
    else:
        choice = st.selectbox("trace", traces, format_func=lambda p: p.stem)
        d = np.load(choice)
        dt = float(d["dt"])
        names = sorted({k.split("__")[0] for k in d.files if "__" in k})
        picked = st.multiselect("controllers", names, default=names)
        faults = json.loads(str(d["faults"]))
        st.caption(", ".join(f"fan {f['fan']}: {f['kind']} at {f['onset'] * dt / 3600:.1f} h" for f in faults)
                   or "no faults in this day")

        def frame(key):
            cols = {n: d[f"{n}__{key}"] for n in picked}
            df = pd.DataFrame(cols)
            df.index = df.index * dt / 3600
            df.index.name = "hours"
            return df

        if picked:
            st.subheader("hottest zone (C)")
            st.line_chart(frame("max_temp"), height=220)
            st.subheader("fan power (W)")
            st.line_chart(frame("fan_w"), height=220)
            st.subheader("lowest fan health heard")
            st.line_chart(frame("min_health"), height=180)
