import pandas as pd
import streamlit as st

from ga import ALL_TIME_SLOTS, genetic_algorithm, read_csv_to_dict

st.title("TV Programme Scheduling using Genetic Algorithm")

ratings = read_csv_to_dict("program_ratings_modified.csv")
HOURS = [f"{h:02d}:00" for h in ALL_TIME_SLOTS]

# ---- Parameter input ----
st.header("Parameters")
mode = st.radio("Parameter setting", ["Default", "Custom"], horizontal=True)

if mode == "Default":
    co_r, mut_r = 0.80, 0.20
    st.write("CO_R = 0.80, MUT_R = 0.20")
else:
    co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, 0.80, 0.01)
    mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.05, 0.05, 0.01)

# ---- Run GA and display schedule ----
if st.button("Run"):
    # Fixed seed so the same parameters always give the same schedule
    schedule, fitness, _ = genetic_algorithm(
        ratings, crossover_rate=co_r, mutation_rate=mut_r, seed=1)

    st.subheader(f"Schedule (CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f})")
    st.table(pd.DataFrame({
        "Time Slot": HOURS,
        "Program": schedule,
        "Rating": [f"{ratings[p][s]:.1f}" for s, p in enumerate(schedule)],
    }).set_index("Time Slot"))
    st.write(f"Total rating: {fitness:.1f}")
