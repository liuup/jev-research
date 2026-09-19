"""Candidate prompts for the 24 game decision model.

Every action of a state becomes one question with two candidate paths, ``YES`` and
``NO``. The question is always the same event: given that this action is taken now and
the frozen policy keeps playing, does the episode finish on the target?
"""

from .env import OPERATIONS_PER_EPISODE

SUCCESS_CANDIDATES = ("yes", "no")


def candidates(row):
    ids = list(row.get("candidate_ids", SUCCESS_CANDIDATES))
    if len(ids) != 2 or set(ids) != set(SUCCESS_CANDIDATES):
        raise ValueError("Every success question needs exactly yes and no")
    return ids


def step_text(step):
    return f"{step['left']} {step['op']} {step['right']} = {step['result']}"


def question_row(state, action, policy_id, candidate_id="yes"):
    """One frozen-policy success question for a single ``(state, action)`` pair."""
    if candidate_id not in SUCCESS_CANDIDATES:
        raise ValueError("Unknown success candidate")
    return dict(
        row_id=state.puzzle_id,
        puzzle_id=state.puzzle_id,
        state_id=state.state_id(),
        depth=state.depth,
        target=str(state.target),
        remaining=[str(value) for value in state.values],
        step_history=[step_text(step) for step in state.history],
        action_key=action["key"],
        action_text=action_text(state, action),
        policy_id=policy_id,
        candidate_ids=list(SUCCESS_CANDIDATES),
        candidate_id=candidate_id,
    )


def action_text(state, action):
    return f"{action['left']} {action['op']} {action['right']} = {action['value']}"


def serialize(row):
    """Prompt for one candidate path of a 24 game action."""
    candidate_id = row["candidate_id"]
    if candidate_id not in candidates(row):
        raise ValueError("Unknown success candidate")
    history = row["step_history"]
    return (
        f"[STATE]\nTarget: {row['target']}\n"
        f"Remaining: {' '.join(row['remaining'])}\n\n"
        "Steps already taken:\n"
        + ("(none)" if not history else "\n".join(
            f"{index + 1}. {text}" for index, text in enumerate(history)
        ))
        + f"\n\nFrozenPolicy: {row['policy_id']}\n\n"
        f"[ACTION]\n{row['action_text']}\n\n"
        "[QUESTION]\nIf this action is performed now and the frozen policy continues, "
        f"will the final value equal {row['target']} after "
        f"{OPERATIONS_PER_EPISODE - row['depth']} more "
        f"{'operation' if OPERATIONS_PER_EPISODE - row['depth'] == 1 else 'operations'}?\n\n"
        "[OPTIONS]\nYES\nNO\n\n"
        f"[CANDIDATE]\n{candidate_id.upper()}"
    )
