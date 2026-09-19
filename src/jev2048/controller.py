"""Greedy controllers built from outcome models; utilities are not action probabilities."""

import numpy as np

from .env import moved
from .evaluation import LOG_TILE_UTILITIES, predict
from .serialization import action_questions


def analyze_reach_2048(model, tokenizer, game, continuation_policy_id,
                       max_length=512, precision="fp32"):
    """All legal actions in one backbone batch; requires a binary-trained model."""
    rows = action_questions(game, continuation_policy_id)
    probabilities = predict(model, tokenizer, rows, microbatch=len(rows),
                            max_length=max_length, precision=precision)
    return {
        row["action"]: {candidate: float(value) for candidate, value in
                        zip(row["candidate_ids"], distribution)}
        for row, distribution in zip(rows, probabilities)
    }


def outcome_utility(p, utility="threshold", threshold=2048):
    if utility == "threshold":
        if threshold not in (256, 512, 1024, 2048, 4096, 8192):
            raise ValueError(threshold)
        return p[:, np.array([128, 256, 512, 1024, 2048, 4096, 8192]) >= threshold].sum(
            -1
        )
    if utility == "log_tile":
        return p @ LOG_TILE_UTILITIES
    raise ValueError(utility)


class JevController:
    def __init__(
        self,
        model,
        tokenizer,
        utility="threshold",
        threshold=2048,
        max_length=512,
        inference_questions=16,
        inference_precision="fp32",
        task="terminal_max_tile",
        continuation_policy_id=None,
    ):
        self.model, self.tokenizer = model, tokenizer
        self.utility, self.threshold, self.max_length = utility, threshold, max_length
        self.inference_questions = inference_questions
        self.inference_precision = inference_precision
        self.task = task
        self.continuation_policy_id = continuation_policy_id

    def choose(self, game):
        return self.choose_many([game])[0]

    def choose_many(self, games):
        if not games:
            return []
        if self.task == "reach_2048":
            if self.utility != "threshold" or self.threshold != 2048:
                raise ValueError("Binary model only supports reaching 2048")
            rows = [row for i, game in enumerate(games) for row in
                    action_questions(game, self.continuation_policy_id, f"actor:{i}:{game.steps}")]
            p = predict(self.model, self.tokenizer, rows, max(4, self.inference_questions),
                        self.max_length, precision=self.inference_precision)
            actions, offset = [], 0
            for game in games:
                legal = game.legal_actions
                actions.append(legal[int(p[offset:offset+len(legal), 0].argmax())])
                offset += len(legal)
            return actions
        legal = [game.legal_actions for game in games]
        if any(not actions for actions in legal):
            raise ValueError("Cannot act in a terminal game")
        rows = [
            dict(**game.state(), action=a, state_id=f"actor:{i}:{game.steps}")
            for i, (game, actions) in enumerate(zip(games, legal))
            for a in actions
        ]
        p = predict(
            self.model, self.tokenizer, rows, self.inference_questions, self.max_length,
            precision=self.inference_precision,
        )
        values = outcome_utility(p, self.utility, self.threshold)
        result, offset = [], 0
        for actions in legal:
            result.append(actions[int(values[offset : offset + len(actions)].argmax())])
            offset += len(actions)
        return result


class HeuristicJudgeController:
    """Code plans with the heuristic; the model judges only near-equal candidates.

    ``delta`` bounds the heuristic value the judge may give up, so delta=0 keeps every
    decision the heuristic would have made alone and only exact ties reach the model.
    ``margin`` requires the model's preferred candidate to beat the heuristic choice by
    that much predicted probability. Both bounds are counted, never assumed.
    """

    def __init__(
        self,
        model,
        tokenizer,
        heuristic,
        delta,
        margin=0.0,
        max_length=512,
        inference_questions=16,
        inference_precision="fp32",
        task="reach_2048",
        continuation_policy_id=None,
    ):
        if task != "reach_2048":
            raise ValueError("The heuristic judge implements reach_2048 only")
        if delta < 0 or margin < 0:
            raise ValueError("delta and margin must be nonnegative")
        if not continuation_policy_id:
            raise ValueError("A frozen continuation policy identifier is required")
        self.model, self.tokenizer, self.heuristic = model, tokenizer, heuristic
        self.delta, self.margin = delta, margin
        self.max_length, self.inference_questions = max_length, inference_questions
        self.inference_precision = inference_precision
        self.continuation_policy_id = continuation_policy_id
        self.counters = dict(
            decisions=0,
            single_legal_action=0,
            code_decisions=0,
            model_decisions=0,
            model_agreements=0,
            margin_overrides=0,
            candidate_count=0,
            queried_candidates=0,
            value_sacrifice=0.0,
            chosen_probability=0.0,
        )

    def candidate_set(self, game):
        values = {
            action: self.heuristic.value(*moved(game.board, action))
            for action in game.legal_actions
        }
        best = max(values.values())
        candidates = [
            action
            for action in game.legal_actions
            if values[action] >= best - self.delta
        ]
        return values, best, candidates

    def choose_many(self, games):
        if not games:
            return []
        plans, rows = [], []
        for index, game in enumerate(games):
            if game.terminal or game.max_tile >= 2048:
                raise ValueError("Cannot judge a terminal or already successful game")
            values, best, candidates = self.candidate_set(game)
            plan = dict(game=game, values=values, best=best, candidates=candidates)
            legal = game.legal_actions
            if len(legal) == 1:
                plan["action"] = legal[0]
                self.counters["single_legal_action"] += 1
            elif len(candidates) == 1:
                plan["action"] = candidates[0]
                self.counters["code_decisions"] += 1
            else:
                wanted = set(candidates)
                questions = [
                    row
                    for row in action_questions(
                        game, self.continuation_policy_id, f"judge:{index}:{game.steps}"
                    )
                    if row["action"] in wanted
                ]
                if len(questions) != len(candidates):
                    raise ValueError("Candidate questions do not match the candidate set")
                plan["questions"] = questions
                rows.extend(questions)
            plans.append(plan)
        if rows:
            probabilities = predict(
                self.model,
                self.tokenizer,
                rows,
                max(4, self.inference_questions),
                self.max_length,
                precision=self.inference_precision,
            )
            offset = 0
            for plan in plans:
                questions = plan.get("questions")
                if not questions:
                    continue
                chunk = probabilities[offset : offset + len(questions)]
                offset += len(questions)
                plan["probabilities"] = {
                    row["action"]: float(value)
                    for row, value in zip(questions, chunk[:, 0])
                }
        return [self.decide(plan) for plan in plans]

    def choose(self, game):
        return self.choose_many([game])[0]

    def decide(self, plan):
        counters = self.counters
        counters["decisions"] += 1
        counters["candidate_count"] += len(plan["candidates"])
        if "action" in plan:
            action = plan["action"]
        else:
            probabilities = plan["probabilities"]
            heuristic_action = self.heuristic.choose(plan["game"])
            model_action = max(plan["candidates"], key=probabilities.__getitem__)
            counters["model_decisions"] += 1
            counters["queried_candidates"] += len(plan["candidates"])
            if probabilities[model_action] > probabilities[heuristic_action] + self.margin:
                action = model_action
            else:
                action = heuristic_action
                counters["margin_overrides"] += 1
            counters["model_agreements"] += int(action == heuristic_action)
            counters["chosen_probability"] += probabilities[action]
        counters["value_sacrifice"] += plan["best"] - plan["values"][action]
        return action

    def summary(self):
        counters = self.counters
        decisions, model = counters["decisions"], counters["model_decisions"]
        return dict(
            decisions=decisions,
            single_legal_action=counters["single_legal_action"],
            code_decisions=counters["code_decisions"],
            model_decisions=model,
            model_agreement_rate=counters["model_agreements"] / model if model else None,
            margin_overrides=counters["margin_overrides"],
            mean_candidate_set_size=counters["candidate_count"] / decisions
            if decisions
            else None,
            mean_queried_candidates=counters["queried_candidates"] / model if model else None,
            mean_heuristic_value_sacrifice=counters["value_sacrifice"] / decisions
            if decisions
            else None,
            mean_chosen_probability=counters["chosen_probability"] / model if model else None,
        )
