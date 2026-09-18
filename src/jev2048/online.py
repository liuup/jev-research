"""Fresh terminal feedback under one frozen policy per generation. No cross-policy replay."""

import hashlib
import multiprocessing as mp
import queue
import random
import time
import traceback
from collections import deque

import numpy as np

from .env import Game
from .serialization import OUTCOMES, bucket


def stream_seed(seed, *parts):
    message = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big") % (
        2**63 - 1
    )


def validate_events(rows, policy_id, generation, stream="train"):
    if not rows:
        raise ValueError("Empty feedback batch")
    for row in rows:
        if (
            row["policy_id"] != policy_id
            or row["generation"] != generation
            or row["stream"] != stream
        ):
            raise ValueError("Feedback from a different policy, generation or split")
        if "q" in row or row["observed_outcome"] not in OUTCOMES:
            raise ValueError(
                "Training requires a single observed event, never MC probabilities"
            )


def terminal_feedback(questions, policy):
    games = []
    for question in questions:
        if question["policy_id"] != policy.policy_id:
            raise ValueError("Rollout policy version mismatch")
        base = Game(
            board=question["board"], score=question["score"], steps=question["steps"]
        )
        game = base.clone()
        game.rng.seed(question["rollout_seed"])
        if not game.step(question["action"]):
            raise ValueError("Illegal counterfactual first action")
        games.append(game)
    active = [i for i, game in enumerate(games) if not game.terminal]
    while active:
        actions = policy.choose_many([games[i] for i in active])
        if len(actions) != len(active):
            raise ValueError("Policy returned wrong number of actions")
        for i, action in zip(active, actions):
            if not games[i].step(action):
                raise ValueError("Continuation policy chose an illegal action")
        active = [i for i in active if not games[i].terminal]
    return [
        dict(
            question,
            observed_outcome=bucket(game.max_tile),
            terminal_score=game.score,
            terminal_max_tile=game.max_tile,
            rollout_steps=game.steps - question["steps"],
        )
        for question, game in zip(questions, games)
    ]


class OnlineCollector:
    def __init__(self, policy, config, generation, stream="train", game_factory=Game):
        self.policy, self.config = policy, config
        self.generation, self.stream = generation, stream
        self.game_factory = game_factory
        self.rng = random.Random(
            stream_seed(config["seed"], generation, stream, "behavior")
        )
        self.serial = 0
        self.games, self.game_ids = [], []
        self.pending = deque()
        self.counters = dict(
            source_games=0, behavior_steps=0, rollout_steps=0, terminal_rollouts=0
        )
        for _ in range(config["source_envs"]):
            self.games.append(None)
            self.game_ids.append(None)
            self._reset(len(self.games) - 1)

    def _reset(self, index):
        serial = self.serial
        self.serial += 1
        seed = stream_seed(
            self.config["seed"], self.generation, self.stream, "source", serial
        )
        self.games[index] = self.game_factory(seed)
        self.game_ids[index] = (
            f"{self.stream}:g{self.generation}:source{serial}:seed{seed}"
        )
        if self.games[index].terminal:
            raise ValueError("Source environment starts terminal")
        self.counters["source_games"] += 1

    def _advance_sources(self):
        actions = self.policy.choose_many(self.games)
        if len(actions) != len(self.games):
            raise ValueError("Wrong number of behavior actions")
        for index, (game, action) in enumerate(zip(self.games, actions)):
            if self.rng.random() < self.config["behavior_epsilon"]:
                action = self.rng.choice(game.legal_actions)
            if not game.step(action):
                raise ValueError("Illegal behavior action")
            self.counters["behavior_steps"] += 1
            if game.terminal:
                self._reset(index)

    def collect_questions(self, minimum):
        questions = []
        while len(questions) < minimum:
            for game, game_id in zip(self.games, self.game_ids):
                if game.steps % self.config["select_every"] == 0:
                    state_id = f"{game_id}:step{game.steps}"
                    for action in game.legal_actions:
                        questions.append(
                            dict(
                                **game.state(),
                                state_id=state_id,
                                source_game=game_id,
                                action=action,
                                policy_id=self.policy.policy_id,
                                generation=self.generation,
                                stream=self.stream,
                                rollout_seed=stream_seed(
                                    self.config["seed"], state_id, action, "event"
                                ),
                            )
                        )
            self._advance_sources()
        return questions

    def collect(self, count):
        while len(self.pending) < count:
            questions = self.collect_questions(count - len(self.pending))
            events = terminal_feedback(questions, self.policy)
            self.counters["terminal_rollouts"] += len(events)
            self.counters["rollout_steps"] += sum(
                row["rollout_steps"] for row in events
            )
            self.pending.extend(events)
        rows = [self.pending.popleft() for _ in range(count)]
        validate_events(rows, self.policy.policy_id, self.generation, self.stream)
        return rows

    def close(self):
        return None


def _prefetch_worker(policy, config, generation, stream, batch_size, output, stop):
    try:
        collector = OnlineCollector(policy, config, generation, stream, Game)
        batch_index = 0
        while not stop.is_set():
            started = time.monotonic()
            rows = collector.collect(batch_size)
            message = dict(
                batch_index=batch_index,
                rows=rows,
                counters=dict(collector.counters),
                producer_seconds=time.monotonic() - started,
            )
            while not stop.is_set():
                try:
                    output.put(message, timeout=0.1)
                    break
                except queue.Full:
                    continue
            batch_index += 1
    except BaseException:
        message = dict(error=traceback.format_exc())
        while not stop.is_set():
            try:
                output.put(message, timeout=0.1)
                break
            except queue.Full:
                continue


class AsyncOnlineCollector:
    """Ordered bounded prefetch for a frozen CPU continuation policy."""

    def __init__(
        self, policy, config, generation, batch_size, prefetch_batches, stream="train"
    ):
        if policy.spec.get("kind") != "heuristic":
            raise ValueError("Async collection currently requires a CPU heuristic policy")
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
        context = mp.get_context("spawn")
        self.output = context.Queue(maxsize=prefetch_batches)
        self.stop = context.Event()
        self.process = context.Process(
            target=_prefetch_worker,
            args=(
                policy,
                config,
                generation,
                stream,
                batch_size,
                self.output,
                self.stop,
            ),
            name=f"jev-prefetch-g{generation}",
        )
        self.policy_id = policy.policy_id
        self.generation = generation
        self.stream = stream
        self.next_batch = 0
        self.counters = dict(
            source_games=0, behavior_steps=0, rollout_steps=0, terminal_rollouts=0
        )
        self.producer_seconds = 0.0
        self.process.start()

    def collect(self, count):
        if not self.process.is_alive() and self.output.empty():
            raise RuntimeError("Online prefetch process exited unexpectedly")
        try:
            message = self.output.get(timeout=300)
        except queue.Empty as error:
            raise RuntimeError("Timed out waiting for online feedback") from error
        if "error" in message:
            raise RuntimeError("Online prefetch failed:\n" + message["error"])
        if message["batch_index"] != self.next_batch or len(message["rows"]) != count:
            raise RuntimeError("Out-of-order or incorrectly sized prefetched batch")
        self.next_batch += 1
        self.counters = message["counters"]
        self.producer_seconds = message["producer_seconds"]
        validate_events(
            message["rows"], self.policy_id, self.generation, self.stream
        )
        return message["rows"]

    def close(self):
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.output.close()
        self.output.join_thread()


def retarget_questions(questions, policy_id, generation, seed, stream):
    return [
        dict(
            {
                k: v
                for k, v in row.items()
                if k
                not in (
                    "observed_outcome",
                    "q",
                    "counts",
                    "rollout_steps",
                    "terminal_score",
                    "terminal_max_tile",
                )
            },
            policy_id=policy_id,
            generation=generation,
            stream=stream,
            rollout_seed=stream_seed(
                seed, generation, stream, row["state_id"], row["action"]
            ),
        )
        for row in questions
    ]


def mc_reference(questions, policy, repeats, seed, rollout_batch=32):
    if repeats < 2 or rollout_batch < 1:
        raise ValueError("MC requires >=2 repeats and positive batch size")
    counts = np.zeros((len(questions), len(OUTCOMES)), dtype=int)
    pending, indices = [], []
    environment_steps = 0
    for index, row in enumerate(questions):
        if "observed_outcome" in row:
            raise ValueError("MC input must be unlabeled evaluation questions")
        for repeat in range(repeats):
            pending.append(
                dict(
                    row,
                    rollout_seed=stream_seed(
                        seed, row["state_id"], row["action"], repeat
                    ),
                )
            )
            indices.append(index)
            if len(pending) == rollout_batch or (
                index == len(questions) - 1 and repeat == repeats - 1
            ):
                for i, event in zip(indices, terminal_feedback(pending, policy)):
                    counts[i, OUTCOMES.index(event["observed_outcome"])] += 1
                    environment_steps += event["rollout_steps"]
                pending, indices = [], []
    return [
        dict(
            row,
            counts=count.tolist(),
            q=(count / repeats).tolist(),
            rollouts=repeats,
            mc_seed=seed,
        )
        for row, count in zip(questions, counts)
    ], environment_steps
