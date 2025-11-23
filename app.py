import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from engine import BluffBotAI, init_game, player_turn, MAX_ROUNDS

st.set_page_config(page_title="BluffBot MVP", page_icon="🃏", layout="wide")

# -----------------------------
# Session state setup
# -----------------------------
if "ai" not in st.session_state:
    st.session_state.ai = BluffBotAI()

if "game" not in st.session_state:
    st.session_state.game = init_game(st.session_state.ai)

g = st.session_state.game
ai = st.session_state.ai

# -----------------------------
# Header
# -----------------------------
st.title("🃏 BluffBot — Adaptive AI Bluffing Game (MVP)")
st.caption("The AI learns your bluffing style over time. Try to outsmart it!")

# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader(f"Round {min(g.round_num, MAX_ROUNDS)} / {MAX_ROUNDS}")
    st.write("### Your Hand")
    st.write(" ".join([f"`{c}`" for c in g.player.hand]) if g.player.hand else "_empty_")

    st.write("### Bot Hand")
    st.write("`Hidden cards`" if g.bot.hand else "_empty_")

    st.write("---")
    st.write("### Make a Move")

    claim = st.selectbox("What do you *claim* you're playing?", ["low", "medium", "high"])
    bluff = st.toggle("Bluff? (play weak card but claim higher)", value=False)

    if st.button("Play Turn", disabled=g.game_over):
        player_turn(g, ai, claim, bluff)
        st.rerun()

    st.info(g.last_message)

    if g.game_over:
        st.success("Game finished!")
        st.write(f"**Final Score — You: {g.player.score} | BluffBot: {g.bot.score}**")
        if g.player.score > g.bot.score:
            st.balloons()
            st.write("🏆 You outbluffed the AI!")
        elif g.player.score < g.bot.score:
            st.write("🤖 BluffBot wins — it learned your patterns!")
        else:
            st.write("🤝 Tie game!")

        if st.button("Start New Game"):
            st.session_state.game = init_game(ai)
            st.rerun()

with right:
    st.subheader("📊 Live Stats")

    st.metric("Your Score", g.player.score)
    st.metric("Bot Score", g.bot.score)

    # Suspicion meter = average bot call tendency for your contexts
    if len(g.history) > 0:
        player_rows = g.history[g.history["actor"] == "player"]
        bluff_rate = player_rows["did_bluff"].mean() if len(player_rows) else 0
        st.progress(min(1.0, bluff_rate))
        st.caption(f"AI Suspicion Meter (your bluff rate): **{int(bluff_rate*100)}%**")
    else:
        st.progress(0.0)
        st.caption("AI Suspicion Meter: 0%")

    st.write("---")
    st.write("### Your Bluff History")

    if len(g.history) > 0:
        player_rows = g.history[g.history["actor"] == "player"].copy()
        player_rows["round"] = player_rows["round"].astype(int)

        fig = plt.figure()
        plt.plot(player_rows["round"], player_rows["did_bluff"], marker="o")
        plt.yticks([0, 1], ["Honest", "Bluff"])
        plt.xlabel("Round")
        plt.ylabel("Move Type")
        plt.title("Bluff Timeline")
        st.pyplot(fig)
    else:
        st.write("_No moves yet_")

# -----------------------------
# History table + download
# -----------------------------
st.write("---")
st.subheader("🧾 Game Log")

st.dataframe(g.history, use_container_width=True)

csv = g.history.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Session Log (CSV)",
    data=csv,
    file_name="bluffbot_session_log.csv",
    mime="text/csv",
)
