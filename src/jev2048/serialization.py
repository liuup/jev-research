from .env import moved


OUTCOMES = ("lt256", "256", "512", "1024", "2048", "4096", "8192_plus")
DESCRIPTIONS = (
    "less than 256",
    "256",
    "512",
    "1024",
    "2048",
    "4096",
    "8192 or greater",
)

REACH_2048 = "reach_2048"
SUCCESS_CANDIDATES = ("yes", "no")


def candidates(row):
    if row.get("task") == REACH_2048:
        ids = list(row.get("candidate_ids", SUCCESS_CANDIDATES))
        if len(ids) != 2 or set(ids) != set(SUCCESS_CANDIDATES):
            raise ValueError("reach_2048 requires exactly yes and no")
        return ids
    return list(row.get("candidate_ids", OUTCOMES))


def action_questions(game, continuation_policy_id, state_id="query:0"):
    """One complete binary question per legal action, in deterministic order."""
    if not continuation_policy_id:
        raise ValueError("A frozen continuation policy identifier is required")
    if game.max_tile >= 2048 or game.terminal:
        raise ValueError("Query requires an unfinished game below 2048")
    return [
        dict(
            **game.state(), task=REACH_2048, action=action, state_id=state_id,
            continuation_policy_id=continuation_policy_id,
            candidate_ids=list(SUCCESS_CANDIDATES),
        )
        for action in game.legal_actions
    ]


def bucket(tile):
    if tile < 256:
        return "lt256"
    if tile >= 8192:
        return "8192_plus"
    return str(tile)


def serialize(row, candidate_id):
    board = "\n".join(" ".join(map(str, r)) for r in row["board"])
    afterstate, merge_score = moved(row["board"], row["action"])
    afterstate_text = "\n".join(" ".join(map(str, r)) for r in afterstate)
    if row.get("task") == REACH_2048:
        if candidate_id not in candidates(row):
            raise ValueError("Unknown success candidate")
        if max(map(max, row["board"])) >= 2048:
            raise ValueError("Already-successful boards are outside reach_2048")
        if afterstate == row["board"] or tuple(map(tuple, afterstate)) == tuple(map(tuple, row["board"])):
            raise ValueError("Question action must be legal")
        policy_id = row["continuation_policy_id"]
        return (
            f"[STATE]\nGame: 2048\nBoard:\n{board}\nScore: {row['score']}\n"
            f"MaxTile: {max(map(max, row['board']))}\n"
            f"FrozenContinuationPolicy: {policy_id}\n\n"
            f"[ACTION]\n{row['action']}\n"
            f"DeterministicAfterstateBeforeSpawn:\n{afterstate_text}\n"
            f"ImmediateMergeScore: {merge_score}\n\n[QUESTION]\n"
            f"If {row['action']} is taken now and the frozen continuation policy "
            "is followed afterwards, will a tile of at least 2048 be reached "
            "before no legal moves remain? Stop immediately upon success. "
            "Use normal stochastic tile spawning.\n\n[OPTIONS]\nYES\nNO\n\n"
            f"[CANDIDATE]\n{candidate_id.upper()}"
        )
    description = dict(zip(OUTCOMES, DESCRIPTIONS)).get(candidate_id, candidate_id)
    return (
        f"[STATE]\nGame: 2048\nBoard:\n{board}\nScore: {row['score']}\n"
        f"MaxTile: {max(map(max, row['board']))}\n\n[ACTION]\n{row['action']}\n"
        f"DeterministicAfterstateBeforeSpawn:\n{afterstate_text}\n"
        f"ImmediateMergeScore: {merge_score}\n\n[QUESTION]\n"
        f"If {row['action']} is taken now and the frozen continuation policy is followed "
        f"until termination, what will be the terminal maximum tile?\n\n[CANDIDATE]\n{description}"
    )
