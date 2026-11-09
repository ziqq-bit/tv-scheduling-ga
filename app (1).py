import altair as alt
import pandas as pd
import streamlit as st

from ga import ALL_TIME_SLOTS, genetic_algorithm, optimal_fitness, read_csv_to_dict

# ------------------------------------------------------------------ setup
st.set_page_config(page_title="TV Scheduling with GA", page_icon="📺", layout="wide")

CSV_FILE = "program_ratings_modified.csv"
HOURS = [f"{h:02d}:00" for h in ALL_TIME_SLOTS]

# The 8 ratings changed from the original program_ratings.csv
MODIFIED = [
    ("movie_b", 20, 0.4, 0.8), ("live_soccer", 21, 0.3, 0.7),
    ("live_soccer", 22, 0.4, 0.9), ("live_soccer", 23, 0.3, 0.8),
    ("movie_a", 21, 0.5, 0.8), ("movie_a", 22, 0.3, 0.9),
    ("tv_series_a", 21, 0.5, 0.8), ("music_program", 22, 0.5, 0.8),
]

ratings = read_csv_to_dict(CSV_FILE)
PROGRAMS = list(ratings.keys())
BEST_POSSIBLE = round(optimal_fitness(ratings), 2)

# Same colour for each programme in every chart
PROGRAM_COLORS = alt.Scale(domain=PROGRAMS, scheme="tableau10")

# ------------------------------------------------------------------ header
st.title("📺 TV Programme Scheduling using Genetic Algorithm")
st.markdown(
    "Find the best one-day TV schedule (06:00–23:00) that maximises total audience rating. "
    "Set the **crossover rate (CO_R)** and **mutation rate (MUT_R)** for three trials, "
    "run the genetic algorithm, and compare the schedules it produces."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Programmes", len(PROGRAMS))
c2.metric("Time slots", len(HOURS))
c3.metric("Modified ratings", len(MODIFIED))
c4.metric("Best possible rating", f"{BEST_POSSIBLE:.1f}")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ GA settings")
    GEN = st.number_input("Generations", 10, 1000, 100, step=10)
    POP = st.number_input("Population size", 10, 500, 50, step=10)
    SEED = st.number_input("Random seed (0 = random)", 0, 10_000, 1,
                           help="Use 1 to reproduce the results in the report.")
    st.divider()
    st.subheader("How the GA works")
    st.markdown(
        "- **Chromosome:** one programme per hour (18 genes)\n"
        "- **Fitness:** sum of ratings in the schedule\n"
        "- **Selection:** tournament of 3\n"
        "- **Crossover:** one-point, probability CO_R\n"
        "- **Mutation:** each hour changes with probability MUT_R\n"
        "- **Elitism:** best schedule kept every generation"
    )

# ------------------------------------------------------------------ ratings data
tab_data, tab_run = st.tabs(["📊 Ratings data", "🧬 Run trials"])

with tab_data:
    st.subheader("Programme ratings by hour")
    st.caption("Orange boxes mark the 8 ratings changed from the original CSV.")

    long = pd.DataFrame(
        [{"Program": p, "Hour": HOURS[i], "Rating": r}
         for p, values in ratings.items() for i, r in enumerate(values)]
    )
    mods = pd.DataFrame(
        [{"Program": p, "Hour": f"{h:02d}:00", "Original": o, "Modified": n}
         for p, h, o, n in MODIFIED]
    )

    base = alt.Chart(long).encode(
        x=alt.X("Hour:O", sort=HOURS, title="Time slot"),
        y=alt.Y("Program:N", sort=PROGRAMS, title=None),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Rating:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        title="Rating"),
        tooltip=["Program", "Hour", "Rating"],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("Rating:Q", format=".1f"),
        color=alt.condition("datum.Rating > 0.55", alt.value("white"), alt.value("black")),
    )
    boxes = alt.Chart(mods).mark_rect(fill=None, stroke="#f58518", strokeWidth=3).encode(
        x=alt.X("Hour:O", sort=HOURS), y=alt.Y("Program:N", sort=PROGRAMS),
        tooltip=["Program", "Hour", "Original", "Modified"],
    )
    st.altair_chart((heat + labels + boxes).properties(height=380))

    st.subheader("Changes made to the CSV")
    st.dataframe(mods, hide_index=True)

# ------------------------------------------------------------------ trials
with tab_run:
    st.subheader("Trial parameters")
    st.caption("CO_R range 0–0.95 (default 0.8). MUT_R range 0.01–0.5 (default 0.2).")

    defaults = [(0.80, 0.20), (0.60, 0.05), (0.95, 0.01)]
    trials = []
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            with st.container(border=True):
                st.markdown(f"#### Trial {i + 1}")
                co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, defaults[i][0], 0.01,
                                 key=f"co{i}")
                mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.50, defaults[i][1], 0.01,
                                  key=f"mu{i}",
                                  help="Chance each hour is changed to a random programme. "
                                       "0.01–0.05 recommended; 0.2 is the tutorial default.")
                trials.append((co_r, mut_r))

    if st.button("▶ Run all 3 trials", type="primary"):
        results = []
        with st.spinner("Evolving schedules..."):
            for i, (co_r, mut_r) in enumerate(trials):
                seed = None if SEED == 0 else SEED + i
                schedule, fit, hist = genetic_algorithm(
                    ratings, generations=GEN, population_size=POP,
                    crossover_rate=co_r, mutation_rate=mut_r, seed=seed)
                reached = next((g for g, v in enumerate(hist) if v >= BEST_POSSIBLE - 1e-6), None)
                results.append({"co_r": co_r, "mut_r": mut_r, "schedule": schedule,
                                "fitness": round(fit, 2), "history": hist, "reached": reached})
        st.session_state["results"] = results

    results = st.session_state.get("results")
    if results:
        st.divider()
        st.subheader("Results")

        # Summary cards
        cols = st.columns(3)
        best_idx = max(range(3), key=lambda i: (results[i]["fitness"],
                                                -(results[i]["reached"] or 10**6)))
        for i, (col, r) in enumerate(zip(cols, results)):
            with col:
                with st.container(border=True):
                    title = f"Trial {i + 1}" + ("  🏆" if i == best_idx else "")
                    st.markdown(f"#### {title}")
                    st.caption(f"CO_R = {r['co_r']:.2f} · MUT_R = {r['mut_r']:.2f}")
                    st.metric("Total rating", f"{r['fitness']:.1f}",
                              f"{r['fitness'] - BEST_POSSIBLE:+.1f} vs best possible",
                              delta_color="off" if r["fitness"] >= BEST_POSSIBLE - 1e-6 else "normal")
                    st.caption(f"Best possible reached at generation {r['reached']}"
                               if r["reached"] is not None else "Best possible not reached")

        # Fitness over generations
        st.markdown("#### Best fitness in each generation")
        hist_df = pd.DataFrame(
            [{"Trial": f"Trial {i + 1} (CO_R {r['co_r']:.2f}, MUT_R {r['mut_r']:.2f})",
              "Generation": g, "Fitness": v}
             for i, r in enumerate(results) for g, v in enumerate(r["history"])]
        )
        lines = alt.Chart(hist_df).mark_line(strokeWidth=2.5).encode(
            x=alt.X("Generation:Q"),
            y=alt.Y("Fitness:Q", scale=alt.Scale(zero=False), title="Best fitness"),
            color=alt.Color("Trial:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Trial", "Generation", alt.Tooltip("Fitness:Q", format=".1f")],
        )
        ref = alt.Chart(pd.DataFrame({"y": [BEST_POSSIBLE]})).mark_rule(
            strokeDash=[6, 4], color="gray").encode(y="y:Q")
        ref_label = alt.Chart(pd.DataFrame({"y": [BEST_POSSIBLE],
                                            "t": [f"Best possible {BEST_POSSIBLE:.1f}"]})
                              ).mark_text(align="left", dx=5, dy=-8, color="gray").encode(
            y="y:Q", x=alt.value(0), text="t:N")
        st.altair_chart((lines + ref + ref_label).properties(height=350))

        # Schedule per trial
        st.markdown("#### Schedules")
        tabs = st.tabs([f"Trial {i + 1}" for i in range(3)])
        for i, (tab, r) in enumerate(zip(tabs, results)):
            with tab:
                table = pd.DataFrame({
                    "Time Slot": HOURS,
                    "Program": r["schedule"],
                    "Rating": [ratings[p][s] for s, p in enumerate(r["schedule"])],
                })
                left, right = st.columns([2, 3])
                with left:
                    st.dataframe(
                        table, hide_index=True, height=670,
                        column_config={"Rating": st.column_config.ProgressColumn(
                            "Rating", min_value=0.0, max_value=1.0, format="%.1f")},
                    )
                    st.markdown(f"**Total rating: {r['fitness']:.1f}** "
                                f"({r['fitness'] / BEST_POSSIBLE:.1%} of best possible)")
                with right:
                    bars = alt.Chart(table).mark_bar().encode(
                        x=alt.X("Time Slot:O", sort=HOURS),
                        y=alt.Y("Rating:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Program:N", scale=PROGRAM_COLORS,
                                        legend=alt.Legend(orient="bottom", columns=4)),
                        tooltip=["Time Slot", "Program", "Rating"],
                    )
                    st.altair_chart(bars.properties(height=420, title="Rating of each scheduled slot"))
                    counts = table["Program"].value_counts().rename_axis("Program") \
                        .reset_index(name="Slots")
                    pie = alt.Chart(counts).mark_arc(innerRadius=50).encode(
                        theta="Slots:Q",
                        color=alt.Color("Program:N", scale=PROGRAM_COLORS, legend=None),
                        tooltip=["Program", "Slots"],
                    )
                    st.altair_chart(pie.properties(height=220, title="Airtime share (hours)"))

        # Comparison table
        st.markdown("#### Comparison")
        st.dataframe(pd.DataFrame([{
            "Trial": i + 1, "CO_R": r["co_r"], "MUT_R": r["mut_r"],
            "Total rating": r["fitness"],
            "% of best possible": f"{r['fitness'] / BEST_POSSIBLE:.1%}",
            "Best possible reached at": f"Generation {r['reached']}" if r["reached"] is not None else "Not reached",
        } for i, r in enumerate(results)]), hide_index=True)
    else:
        st.info("Set the parameters above and click **Run all 3 trials** to see the results.")
