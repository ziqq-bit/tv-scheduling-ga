import pandas as pd
import streamlit as st

from ga import ALL_TIME_SLOTS, genetic_algorithm, optimal_fitness, read_csv_to_dict

st.set_page_config(page_title="TV Scheduling with GA", page_icon="📺")

st.title("📺 TV Programme Scheduling using Genetic Algorithm")
st.write("Set the crossover and mutation rates, then click **Run** to generate "
         "the best one-day TV schedule (06:00–23:00).")

ratings = read_csv_to_dict("program_ratings_modified.csv")
BEST_POSSIBLE = optimal_fitness(ratings)
HOURS = [f"{h:02d}:00" for h in ALL_TIME_SLOTS]


def first_reached(history):
    """Generation where the best fitness first reached the best possible total."""
    return next((g for g, v in enumerate(history) if v >= BEST_POSSIBLE - 1e-6), None)


# ---- Parameter input ----
with st.container(border=True):
    st.subheader("⚙️ Parameters")
    mode = st.radio("Parameter setting", ["Default", "Custom"], horizontal=True,
                    help="Default uses CO_R 0.80 and MUT_R 0.20. "
                         "Custom lets you set your own values.")

    if mode == "Default":
        co_r, mut_r = 0.80, 0.20
        c1, c2 = st.columns(2)
        c1.metric("CO_R (Crossover Rate)", f"{co_r:.2f}")
        c2.metric("MUT_R (Mutation Rate)", f"{mut_r:.2f}")
    else:
        c1, c2 = st.columns(2)
        co_r = c1.slider("CO_R (Crossover Rate)", 0.0, 0.95, 0.80, 0.01,
                         help="Range 0 to 0.95")
        mut_r = c2.slider("MUT_R (Mutation Rate)", 0.01, 0.05, 0.05, 0.01,
                          help="Range 0.01 to 0.05")

    run_10 = st.checkbox("Also run 10 times to check the average")
    run = st.button("▶ Run", type="primary")

# ---- Run GA and display results ----
if run:
    # Fixed seed so the same parameters always give the same schedule
    schedule, fitness, history = genetic_algorithm(
        ratings, crossover_rate=co_r, mutation_rate=mut_r, seed=1)
    reached = first_reached(history)

    st.subheader(f"📋 Schedule (CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f})")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total rating", f"{fitness:.1f}", help=f"Best possible is {BEST_POSSIBLE:.1f}")
    m2.metric("9.7 first reached at", f"Gen {reached}" if reached is not None else "Not reached")
    m3.metric("Rating at generation 20", f"{history[20]:.1f}")

    st.table(pd.DataFrame({
        "Time Slot": HOURS,
        "Program": schedule,
        "Rating": [f"{ratings[p][s]:.1f}" for s, p in enumerate(schedule)],
    }).set_index("Time Slot"))

    st.write("**Best rating in each generation**")
    st.line_chart(pd.DataFrame({"Best rating": history}).rename_axis("Generation"))

    if run_10:
        fits, gens = [], []
        for seed in range(1, 11):
            _, f, h = genetic_algorithm(ratings, crossover_rate=co_r,
                                        mutation_rate=mut_r, seed=seed)
            fits.append(f)
            g = first_reached(h)
            if g is not None:
                gens.append(g)

        st.subheader("🔁 Average of 10 runs (seeds 1 to 10)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Average total rating", f"{sum(fits) / 10:.2f}")
        a2.metric(f"Runs reaching {BEST_POSSIBLE:.1f}", f"{len(gens)} of 10")
        a3.metric("Average generation", f"{sum(gens) / len(gens):.1f}" if gens else "Not reached")
