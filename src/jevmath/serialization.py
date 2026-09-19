"""Candidate prompts and observed-event records for the reasoning environment."""

from .env import SYMBOL

SUCCESS_CANDIDATES = ("yes", "no")


def candidates(row):
    ids = list(row.get("candidate_ids", SUCCESS_CANDIDATES))
    if len(ids) != 2 or set(ids) != set(SUCCESS_CANDIDATES):
        raise ValueError("Every reasoning candidate needs exactly yes and no")
    return ids


def known_quantities(state):
    return "\n".join(
        f"{label} = {value}" for value, label in state.quantities
    )


def steps_text(state):
    if not state.steps:
        return "(none)"
    return "\n".join(
        f"{index + 1}. {step['left']} {SYMBOL[step['op']]} {step['right']} = {step['result']}"
        for index, step in enumerate(state.steps)
    )


def question_row(state, candidate, policy_id, candidate_id="yes"):
    """One observed-event question for a single (state, candidate action) pair."""
    if candidate_id not in SUCCESS_CANDIDATES:
        raise ValueError("Unknown success candidate")
    return dict(
        row_id=state.row["id"],
        state_id=state.state_id(),
        depth=state.depth,
        question=state.question,
        gold=str(state.gold),
        pool=[(str(value), label) for value, label in state.quantities],
        step_history=list(state.steps),
        action_key=candidate["key"],
        action_kind=candidate["kind"],
        action_text=state.describe(candidate),
        policy_id=policy_id,
        candidate_ids=list(SUCCESS_CANDIDATES),
        candidate_id=candidate_id,
    )


def serialize(row):
    """Prompt for one candidate path; the question is the frozen-policy success event."""
    candidate_id = row["candidate_id"]
    if candidate_id not in candidates(row):
        raise ValueError("Unknown success candidate")
    return (
        f"[STATE]\nArithmetic word problem:\n{row['question']}\n\n"
        f"Known quantities:\n"
        + "\n".join(f"{label} = {value}" for value, label in row["pool"])
        + f"\n\nSteps already taken:\n"
        + (
            "(none)"
            if not row["step_history"]
            else "\n".join(
                f"{index + 1}. {step['left']} {SYMBOL[step['op']]} {step['right']} = {step['result']}"
                for index, step in enumerate(row["step_history"])
            )
        )
        + f"\n\nFrozenContinuationPolicy: {row['policy_id']}\n\n"
        f"[ACTION]\n{row['action_text']}\n\n"
        "[QUESTION]\nIf this action is taken now and the frozen continuation policy is "
        "followed afterwards, will the reported final answer be correct? Use the "
        "calculator results shown above.\n\n"
        "[OPTIONS]\nYES\nNO\n\n"
        f"[CANDIDATE]\n{candidate_id.upper()}"
    )
