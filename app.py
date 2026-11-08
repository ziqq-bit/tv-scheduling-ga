import pandas as pd
import streamlit as st

from ga import (ALL_TIME_SLOTS, genetic_algorithm, optimal_fitness,
                read_csv_to_dict)

st.set_page_config(page_title="TV Scheduling with GA", layout="wide")
st.title("📺 TV Programme Scheduling using Genetic Algorithm")
st.caption("JIE42903 Evolutionary Computing – Assignment")

ratings = read_csv_to_dict("program_ratings_modified.csv")

with st.expander("View programme ratings (modified CSV)"):
    df_ratings = pd.DataFrame(ratings, index=[f"{h:02d}:00" for h in ALL_TIME_SLOTS]).T
    st.dataframe(df_ratings, use_container_width=True)

# ---- Fixed GA settings ----
st.sidebar.header("Fixed GA settings")
GEN = st.sidebar.number_input("Generations", 10, 1000, 100, step=10)
POP = st.sidebar.number_input("Population size", 10, 500, 50, step=10)

SEED = st.sidebar.number_input("Random seed (0 = random)", 0, 10_000, 0)

# ---- Trial parameters ----
st.header("Trial parameters")
st.write("Set the crossover rate (CO_R) and mutation rate (MUT_R) for each of the three trials.")

defaults = [(0.80, 0.20), (0.60, 0.05), (0.95, 0.01)]
trials = []
cols = st.columns(3)
for i, col in enumerate(cols):
    with col:
        st.subheader(f"Trial {i + 1}")
        co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, defaults[i][0], 0.01, key=f"co{i}")
        mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.50, defaults[i][1], 0.01, key=f"mu{i}",
                          help="Chance each hour is changed to a random programme. 0.01-0.05 recommended; 0.2 is the tutorial default")
        trials.append((co_r, mut_r))

if st.button("▶ Run all 3 trials", type="primary"):
    best_possible = optimal_fitness(ratings)
    summary, histories = [], {}

    for i, (co_r, mut_r) in enumerate(trials):
        seed = None if SEED == 0 else SEED + i
        schedule, fit, hist = genetic_algorithm(
            ratings, generations=GEN, population_size=POP,
            crossover_rate=co_r, mutation_rate=mut_r,
            seed=seed)

        st.subheader(f"Trial {i + 1}: CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f}")
        table = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in ALL_TIME_SLOTS],
            "Program": schedule,
            "Rating": [ratings[p][s] for s, p in enumerate(schedule)],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.write(f"**Total rating (fitness): {fit:.2f}** "
                 f"(best possible {best_possible:.2f}, {fit / best_possible:.1%})")

        summary.append({"Trial": i + 1, "CO_R": co_r, "MUT_R": mut_r,
                        "Total Rating": round(fit, 2),
                        "% of optimum": f"{fit / best_possible:.1%}"})
        histories[f"Trial {i + 1}"] = hist

    st.header("Comparison")
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.write("Best fitness per generation")
    st.line_chart(pd.DataFrame(histories))
