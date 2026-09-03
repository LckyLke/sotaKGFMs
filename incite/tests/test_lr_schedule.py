"""make_lr_schedule: steps before first_step run at the base lr (the
from-scratch recipe run passes --schedule_start 20001 to get 20k constant
steps and then a warmup plus linear decay over the last 10k)."""

from incite.pretrain import make_lr_schedule


def test_steps_before_the_schedule_start_run_at_the_base_lr():
    lr_at = make_lr_schedule(1.0, "linear", 0.0, 500, 20001, 30000)
    assert lr_at(1) == 1.0 and lr_at(20000) == 1.0
    assert abs(lr_at(20001) - 1.0 / 500) < 1e-9
    assert abs(lr_at(20500) - 1.0) < 1e-9
    assert abs(lr_at(30000)) < 1e-6
    # a continuation's own first step keeps its warmup
    cont = make_lr_schedule(1.0, "linear", 0.0, 500, 20001, 30000)
    assert cont(20001) < cont(20400) < cont(20500)
    const = make_lr_schedule(2.0, "constant", 0.0, 0, 1, 100)
    assert const(1) == 2.0 and const(100) == 2.0
