from ml.bayesian import evaluate_model, load_matches


def test_match_data_loads() -> None:
    matches = load_matches()
    assert not matches.empty
    assert {"HomeTeam", "AwayTeam", "FTR"} <= set(matches.columns)


def test_bayesian_accuracy_exceeds_threshold() -> None:
    report, _ = evaluate_model()
    assert report["accuracy"] > 0.60
    assert report["train_rows"] > 0
    assert report["test_rows"] > 0
