from jev2048.trainer import learning_rate_scale, question_schedule


def test_continuation_question_schedule_is_exact_suffix():
    complete = question_schedule(11, 9, 4, 17)
    continued = question_schedule(11, 6, 4, 17, global_step_offset=3)
    assert continued == complete[3 * 4 :]


def test_continuation_learning_rate_schedule():
    assert learning_rate_scale(0, 5000, 50) == 1 / 50
    assert learning_rate_scale(49, 5000, 50) == 1
    assert learning_rate_scale(50, 5000, 50) == 1
    assert 0 < learning_rate_scale(4999, 5000, 50) < 1
    assert learning_rate_scale(5000, 5000, 50) == 0
