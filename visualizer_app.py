"""
Interactive visualization dashboard for training results.

Run with: streamlit run visualizer_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(page_title="RL Training Visualizer", layout="wide")

st.title("Q-Learning & SARSA Grid World - Training Visualizer")

# Sidebar info
st.sidebar.markdown("""
## Navigation
Use tabs below to explore different visualizations.

### Plots Directory
All generated plots are in `/plots/`
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Training Progression",
    "Statistical Analysis",
    "Heatmaps",
    "Final Policies",
    "Summary"
])

plots_dir = "plots"

# Helper function to display image
def display_plot(filename, title=""):
    filepath = os.path.join(plots_dir, filename)
    if os.path.exists(filepath):
        st.subheader(title if title else filename.replace('_', ' ').replace('.png', ''))
        image = Image.open(filepath)
        st.image(image, use_column_width=True)
    else:
        st.error(f"Plot not found: {filename}")

# ============ TAB 1: TRAINING PROGRESSION ============
with tab1:
    st.markdown("""
    ## Training Progression

    Watch how the agents learn over time. Each subplot shows a test episode
    at different training stages (episodes 0, 5, 10, 20, 40, 80).

    - **Green circles** = steps agent took
    - **Arrows** = direction of movement
    - **Red dot** = start, **Cyan dot** = goal
    - **Reward** shown at top of each subplot
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Q-Learning Progression")
        display_plot("task_e_q_progression.png", "Q-Learning: Episodes 0→5→10→20→40→80")

    with col2:
        st.markdown("### SARSA Progression")
        display_plot("task_e_sarsa_progression.png", "SARSA: Episodes 0→5→10→20→40→80")

    st.info("""
    **What to look for:**
    - Early episodes (0-10): Agent wanders, negative rewards, many steps
    - Middle episodes (20-40): Agent starts finding paths, avoiding obstacles
    - Late episodes (80+): Efficient paths, high rewards, fewer steps
    """)

# ============ TAB 2: STATISTICAL ANALYSIS ============
with tab2:
    st.markdown("""
    ## Statistical Analysis

    Learning curves, convergence rates, and algorithm comparison.
    """)

    # Learning rate comparison
    st.subheader("Learning Rate Comparison (Task C)")
    st.markdown("4 different alpha values (1.0, 0.5, 0.1, 0.01) trained on the same environment.")
    display_plot("task_c_alpha_comparison.png")

    st.divider()

    # Learning curves
    st.subheader("Learning Progress (Task E)")
    st.markdown("Q-Learning vs SARSA with best learning rate.")
    display_plot("task_e_learning_curves.png")

    st.divider()

    # Algorithm comparison
    st.subheader("Algorithm Comparison (Task D)")
    st.markdown("Off-policy (Q-Learning) vs On-policy (SARSA)")
    display_plot("task_d_algorithm_comparison.png")

    st.info("""
    **Key insights:**
    - Q-Learning uses max Q-value of next state (off-policy) → faster learning
    - SARSA uses actual next action Q-value (on-policy) → more conservative
    """)

# ============ TAB 3: HEATMAPS ============
with tab3:
    st.markdown("""
    ## Q-Value Heatmaps

    Shows learned Q-values for each action direction.
    Brighter = higher value = better action in that state.
    """)

    action_names = {
        0: "North (Up)",
        1: "South (Down)",
        2: "East (Right)",
        3: "West (Left)"
    }

    cols = st.columns(2)
    for action in range(4):
        col = cols[action % 2]
        with col:
            filename = f"bonus_qheatmap_action_{action}.png"
            display_plot(filename, f"Action {action}: {action_names[action]}")

    st.info("""
    **How to read:**
    - Red = low Q-values (avoid these state-action pairs)
    - Yellow = medium Q-values
    - Green = high Q-values (good actions to take)
    - Each cell is a state in the 5x5 grid
    """)

# ============ TAB 4: FINAL POLICIES ============
with tab4:
    st.markdown("""
    ## Final Learned Policies

    Grid visualization with state values and optimal policy directions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Q-Learning Final Policy")
        st.markdown("Arrows show greedy policy. Colors show state values.")
        display_plot("task_f_gridworld_qlearning.png")

    with col2:
        st.subheader("SARSA Final Policy")
        st.markdown("Arrows show greedy policy. Colors show state values.")
        display_plot("task_f_gridworld_sarsa.png")

    st.info("""
    **Color scale:**
    - Green = high state value (close to goal)
    - Yellow = medium state value
    - Red = low state value (far from goal or dead-ends)

    **Arrows:**
    - N = North, S = South, E = East, W = West
    - Shows the greedy policy (best action) in each cell
    """)

# ============ TAB 5: SUMMARY ============
with tab5:
    st.markdown("""
    ## Summary

    ### What This Project Demonstrates

    #### Task (A): Exploration vs Exploitation
    - **Exploration**: Epsilon-greedy strategy (random vs greedy)
    - **Exploitation**: Greedy action selection based on learned Q-values
    - Epsilon decay: start high (explore), decrease over time (exploit)

    #### Task (B): RL Formulation
    - **State**: Grid position (0-24, 5x5 grid)
    - **Action**: Direction (North, South, East, West)
    - **Reward**: +10 (goal), +5 (jump), -1 (step/obstacle)
    - **Policy**: Mapping from states to actions

    #### Task (C): Learning Rate Effects
    - Tested α ∈ [1.0, 0.5, 0.1, 0.01]
    - Higher α: noisier but faster initial learning
    - Lower α: smoother learning, more stable

    #### Task (D): Q-Learning vs SARSA
    - **Q-Learning**: Off-policy, bootstraps from max Q-value → aggressive
    - **SARSA**: On-policy, bootstraps from actual next action → conservative
    - Q-Learning typically converges faster

    #### Task (E): Training & Convergence
    - Convergence criteria: avg reward > 10 over 30 consecutive episodes
    - Progression visualizations show learning curve
    - Agent improves dramatically in first 20 episodes

    #### Task (F): State Values & Policy Visualization
    - State values: cumulative expected reward from that state
    - Policy arrows: optimal action direction
    - Clear gradient from goal (high value) to obstacles (low value)

    ### Performance Metrics
    """)

    # Show metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Grid Size", "5×5")
        st.metric("Max Episodes", "100")

    with metrics_col2:
        st.metric("Convergence Threshold", ">10 reward")
        st.metric("Window Size", "30 episodes")

    with metrics_col3:
        st.metric("Actions", "4 (N,S,E,W)")
        st.metric("Agents", "Q-Learning + SARSA")

    st.divider()
    st.markdown("""
    ### Files Generated

    **Main Plots:**
    - `task_c_alpha_comparison.png` - Learning rate sweep
    - `task_e_learning_curves.png` - Training curves
    - `task_e_q_progression.png` - Q-Learning training progression
    - `task_e_sarsa_progression.png` - SARSA training progression
    - `task_d_algorithm_comparison.png` - Algorithm comparison
    - `task_f_gridworld_qlearning.png` - Final Q-Learning policy
    - `task_f_gridworld_sarsa.png` - Final SARSA policy

    **Bonus:**
    - `bonus_qheatmap_action_0-3.png` - Q-value heatmaps for each action
    """)

# Footer
st.divider()
st.markdown("""
---
**COM762 CW2 - Deep Learning and Its Application**
Q-Learning & SARSA Agent in Grid World
Generated with Streamlit
""")
