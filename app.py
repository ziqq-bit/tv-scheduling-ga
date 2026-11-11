import pandas as pd
import streamlit as st

from ga import ALL_TIME_SLOTS, genetic_algorithm, optimal_fitness, read_csv_to_dict

st.title("TV Programme Scheduling using Genetic Algorithm")

ratings = read_csv_to_dict("program_ratings_modified.csv")
BEST_POSSIBLE = optimal_fitness(ratings)
HOURS = [f"{h:02d}:00" for h in ALL_TIME_SLOTS]


def first_reached(history):
    """Generation where the best fitness first reached the best possible total."""
    return next((g for g, v in enumerate(history) if v >= BEST_POSSIBLE - 1e-6), None)


# ---- Parameter input ----
st.header("Parameters")
mode = st.radio("Parameter setting", ["Default", "Custom"], horizontal=True)

if mode == "Default":
    co_r, mut_r = 0.80, 0.20
    st.write("CO_R = 0.80, MUT_R = 0.20")
else:
    co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, 0.80, 0.01)
    mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.05, 0.05, 0.01)

run_10 = st.checkbox("Also run 10 times to check the average")

# ---- Run GA and display results ----
if st.button("Run"):
    # Fixed seed so the same parameters always give the same schedule
    schedule, fitness, history = genetic_algorithm(
        ratings, crossover_rate=co_r, mutation_rate=mut_r, seed=1)
    reached = first_reached(history)

    st.subheader(f"Schedule (CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f})")
    st.table(pd.DataFrame({
        "Time Slot": HOURS,
        "Program": schedule,
        "Rating": [f"{ratings[p][s]:.1f}" for s, p in enumerate(schedule)],
    }).set_index("Time Slot"))

    st.write(f"Total rating: {fitness:.1f} (best possible {BEST_POSSIBLE:.1f})")
    st.write(f"Best possible first reached at generation: "
             f"{reached if reached is not None else 'not reached'}")
    st.write(f"Best rating at generation 20: {history[20]:.1f}")

    st.write("Best rating in each generation")
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
        st.subheader("Average of 10 runs (seeds 1 to 10)")
        st.write(f"Average total rating: {sum(fits) / 10:.2f}")
        st.write(f"Runs that reached {BEST_POSSIBLE:.1f}: {len(gens)} of 10")
        st.write("Average generation to reach it: "
                 + (f"{sum(gens) / len(gens):.1f}" if gens else "not reached"))
