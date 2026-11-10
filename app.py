import pandas as pd
import streamlit as st

from ga import ALL_TIME_SLOTS, genetic_algorithm, optimal_fitness, read_csv_to_dict

st.set_page_config(page_title="TV Scheduling with GA")
st.title("TV Programme Scheduling using Genetic Algorithm")

ratings = read_csv_to_dict("program_ratings_modified.csv")
BEST_POSSIBLE = optimal_fitness(ratings)
HOURS = [f"{h:02d}:00" for h in ALL_TIME_SLOTS]

with st.expander("View programme ratings (modified CSV)"):
    st.dataframe(pd.DataFrame(ratings, index=HOURS).T)

# ---- Parameters ----
st.header("Parameters")
mode = st.radio("Parameter setting", ["Default", "Custom"], horizontal=True)

if mode == "Default":
    co_r, mut_r = 0.80, 0.20
    st.write("Using default values: CO_R = 0.80, MUT_R = 0.20")
else:
    co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, 0.80, 0.01)
    mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.05, 0.05, 0.01)

seed = st.number_input("Random seed (0 = random)", 0, 10_000, 1)

# ---- Run GA and show result ----
if st.button("Run", type="primary"):
    schedule, fitness, history = genetic_algorithm(
        ratings, crossover_rate=co_r, mutation_rate=mut_r,
        seed=None if seed == 0 else seed)
    reached = next((g for g, v in enumerate(history) if v >= BEST_POSSIBLE - 1e-6), None)

    st.subheader(f"Result: CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f}")
    table = pd.DataFrame({
        "Time Slot": HOURS,
        "Program": schedule,
        "Rating": [f"{ratings[p][s]:.1f}" for s, p in enumerate(schedule)],
    }).set_index("Time Slot")
    st.table(table)

    st.write(f"**Total rating: {fitness:.1f}** (best possible {BEST_POSSIBLE:.1f})")
    st.write(f"Best possible reached at generation {reached}" if reached is not None
             else "Best possible not reached")
