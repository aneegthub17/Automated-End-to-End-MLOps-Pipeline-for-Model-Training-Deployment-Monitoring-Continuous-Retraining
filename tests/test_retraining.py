from src.retraining.retrainer import RetrainingManager


# ============================================================
# TEST RETRAINING DECISION LOGIC
# ============================================================

def test_candidate_better_than_production():

    manager = RetrainingManager()

    result = manager.compare_models(
        production_f1=0.40,
        candidate_f1=0.50
    )

    assert result["candidate_better"] is True

    assert (
        "better"
        in result["reason"].lower()
    )


def test_candidate_equal_to_production():

    manager = RetrainingManager()

    result = manager.compare_models(
        production_f1=0.50,
        candidate_f1=0.50
    )

    assert result["candidate_better"] is False

    assert (
        "equal"
        in result["reason"].lower()
    )


def test_candidate_worse_than_production():

    manager = RetrainingManager()

    result = manager.compare_models(
        production_f1=0.50,
        candidate_f1=0.40
    )

    assert result["candidate_better"] is False

    assert (
        "lower"
        in result["reason"].lower()
    )


def test_candidate_promoted_when_no_production_exists():

    manager = RetrainingManager()

    result = manager.compare_models(
        production_f1=None,
        candidate_f1=0.40
    )

    assert result["candidate_better"] is True

    assert (
        "no production"
        in result["reason"].lower()
    )