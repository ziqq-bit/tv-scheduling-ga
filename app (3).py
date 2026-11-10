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
st.header("Trial Parameters")
st.write("CO_R: 0 to 0.95 (default 0.8). MUT_R: default 0.2, then set between 0.01 and 0.05.")
SEED = st.number_input("Random seed (0 = random)", 0, 10_000, 1)

trials = []
cols = st.columns(3)
for i, col in enumerate(cols):
    with col:
        st.subheader(f"Trial {i + 1}")
        co_r = st.slider("CO_R (Crossover Rate)", 0.0, 0.95, 0.80, 0.01, key=f"co{i}")
        mut_r = st.slider("MUT_R (Mutation Rate)", 0.01, 0.20, 0.20, 0.01, key=f"mu{i}")
        trials.append((co_r, mut_r))

# ---- Run GA and show results ----
if st.button("Run all 3 trials", type="primary"):
    summary = []
    for i, (co_r, mut_r) in enumerate(trials):
        seed = None if SEED == 0 else SEED + i
        schedule, fitness, history = genetic_algorithm(
            ratings, crossover_rate=co_r, mutation_rate=mut_r, seed=seed)

        st.subheader(f"Trial {i + 1}: CO_R = {co_r:.2f}, MUT_R = {mut_r:.2f}")
        table = pd.DataFrame({
            "Time Slot": HOURS,
            "Program": schedule,
            "Rating": [f"{ratings[p][s]:.1f}" for s, p in enumerate(schedule)],
        }).set_index("Time Slot")
        st.table(table)
        reached = next((g for g, v in enumerate(history) if v >= BEST_POSSIBLE - 1e-6), None)
        st.write(f"**Total rating: {fitness:.1f}** (best possible {BEST_POSSIBLE:.1f}) · "
                 + (f"best possible reached at generation {reached}" if reached is not None
                    else "best possible not reached"))

        summary.append({"Trial": i + 1, "CO_R": co_r, "MUT_R": mut_r,
                        "Total Rating": f"{fitness:.1f}",
                        "Reached 9.7 at": f"Generation {reached}" if reached is not None else "Not reached"})

    st.header("Comparison")
    st.table(pd.DataFrame(summary).set_index("Trial"))
