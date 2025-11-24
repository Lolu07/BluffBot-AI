import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# Game configuration

HAND_SIZE = 5
MAX_ROUNDS = 10

# Strength scale: 1 (weak) to 10 (strong)
def draw_hand(size=HAND_SIZE):
    return list(np.random.randint(1, 11, size))

def strength_bucket(card_strength):
    if card_strength <= 3:
        return "weak"
    elif card_strength <= 7:
        return "medium"
    return "strong"

def claim_to_strength(claim):
    # claim categories map to numeric expectation
    return {"low": 3, "medium": 6, "high": 9}[claim]

@dataclass
class PlayerState:
    hand: list = field(default_factory=list)
    score: int = 0

@dataclass
class GameState:
    round_num: int = 1
    player: PlayerState = field(default_factory=PlayerState)
    bot: PlayerState = field(default_factory=PlayerState)
    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=[
            "round", "actor", "actual_strength", "claim",
            "did_bluff", "bot_called", "result"
        ]
    ))
    last_message: str = ""
    game_over: bool = False

class BluffBotAI:
    """
    Simple adaptive AI:
    - Tracks your bluff rate by context (actual strength bucket + claim type).
    - Calls bluff more often when your bluff rate is high in that context.
    """
    def __init__(self):
        # key -> (bluff_count, total_count)
        self.context_stats = {}

    def update_stats(self, actual_strength, claim, did_bluff):
        key = (strength_bucket(actual_strength), claim)
        b, t = self.context_stats.get(key, (0, 0))
        self.context_stats[key] = (b + int(did_bluff), t + 1)

    def bluff_probability(self, actual_strength, claim):
        key = (strength_bucket(actual_strength), claim)
        if key not in self.context_stats:
            # prior belief: people bluff ~35% overall
            return 0.35
        b, t = self.context_stats[key]
        # Laplace smoothing to avoid overconfidence early
        return (b + 1) / (t + 3)

    def decide_call(self, actual_strength, claim):
        p_bluff = self.bluff_probability(actual_strength, claim)

        # AI suspicion meter = p_bluff scaled to 0..100
        suspicion = int(p_bluff * 100)

        # Decide to call based on probability + small randomness
        # If p_bluff high, call more often
        threshold = 0.55
        call = (p_bluff + np.random.uniform(-0.05, 0.05)) > threshold
        return call, suspicion

def init_game(ai: BluffBotAI):
    g = GameState()
    g.player.hand = draw_hand()
    g.bot.hand = draw_hand()
    g.last_message = "New game started. Your move!"
    g.game_over = False
    return g

def log_event(g: GameState, actor, actual_strength, claim, did_bluff, bot_called, result):
    row = {
        "round": g.round_num,
        "actor": actor,
        "actual_strength": actual_strength,
        "claim": claim,
        "did_bluff": int(did_bluff),
        "bot_called": int(bot_called),
        "result": result
    }
    g.history = pd.concat([g.history, pd.DataFrame([row])], ignore_index=True)

def player_turn(g: GameState, ai: BluffBotAI, claim: str, bluff: bool):
    # player chooses a card to play: strongest if truthful, weakest if bluffing
    if bluff:
        actual_strength = min(g.player.hand)
    else:
        actual_strength = max(g.player.hand)

    # remove played card
    g.player.hand.remove(actual_strength)

    bot_called, suspicion = ai.decide_call(actual_strength, claim)

    # resolve scoring
    if bot_called:
        if bluff:
            g.bot.score += 1
            result = "Bot caught your bluff (+1 Bot)"
            g.last_message = f"Bot called bluff! ✅ Caught you. (Suspicion {suspicion}%)"
        else:
            g.player.score += 1
            result = "Bot wrongly called bluff (+1 You)"
            g.last_message = f"Bot called bluff! ❌ You were honest. (Suspicion {suspicion}%)"
    else:
        if bluff:
            g.player.score += 1
            result = "Your bluff succeeded (+1 You)"
            g.last_message = f"Bot accepted. 😈 Bluff worked! (Suspicion {suspicion}%)"
        else:
            result = "Honest play (no call)"
            g.last_message = f"Bot accepted. 👍 Honest play. (Suspicion {suspicion}%)"

    # update AI learning
    ai.update_stats(actual_strength, claim, bluff)

    # log
    log_event(g, "player", actual_strength, claim, bluff, bot_called, result)

    # bot also plays a turn (simple mirror)
    bot_act(g, ai)

    # advance round
    g.round_num += 1
    if g.round_num > MAX_ROUNDS or (len(g.player.hand) == 0 and len(g.bot.hand) == 0):
        g.game_over = True
        g.last_message += " | Game Over!"

def bot_act(g: GameState, ai: BluffBotAI):
    # bot randomly decides to bluff 30% of time
    bluff = np.random.rand() < 0.30

    if bluff:
        actual_strength = min(g.bot.hand)
        claim = np.random.choice(["medium", "high"])  # bluffs upward
    else:
        actual_strength = max(g.bot.hand)
        claim = np.random.choice(["low", "medium", "high"])

    g.bot.hand.remove(actual_strength)

    # player doesn't call in MVP; bot gets a point if bluffing and not called
    bot_called = False

    if bluff:
        g.bot.score += 1
        result = "Bot bluff succeeded (+1 Bot)"
    else:
        result = "Bot honest play"

    log_event(g, "bot", actual_strength, claim, bluff, bot_called, result)


