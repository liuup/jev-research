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


def bucket(tile):
    if tile < 256:
        return "lt256"
    if tile >= 8192:
        return "8192_plus"
    return str(tile)


def serialize(row, candidate_id):
    board = "\n".join(" ".join(map(str, r)) for r in row["board"])
    description = dict(zip(OUTCOMES, DESCRIPTIONS)).get(candidate_id, candidate_id)
    return (
        f"[STATE]\nGame: 2048\nBoard:\n{board}\nScore: {row['score']}\n"
        f"MaxTile: {max(map(max, row['board']))}\n\n[QUESTION]\n"
        f"If {row['action']} is taken now and the frozen continuation policy is followed "
        f"until termination, what will be the terminal maximum tile?\n\n[CANDIDATE]\n{description}"
    )
