from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

TRAIN_SEASONS = (
    "s1617",
    "s1718",
    "s1819",
    "s1920",
    "s2021",
    "s2122",
    "s2223",
    "s2324",
)
TEST_SEASONS = ("s2425", "s2526", "s2627")
ODDS_COLUMNS = ("B365H", "B365D", "B365A")


class EvaluationReport(TypedDict):
    accuracy: float
    train_rows: int
    test_rows: int
    classification_report: dict[str, object]


def load_matches(data_dir: str | Path = "data") -> pd.DataFrame:
    root = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(root.glob("*/E0.csv")):
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        columns = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "B365H",
            "B365D",
            "B365A",
        ]
        available = [column for column in columns if column in frame.columns]
        reduced = frame.loc[:, available].copy()
        reduced["season"] = csv_path.parent.name
        reduced["date"] = pd.to_datetime(
            reduced["Date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        frames.append(reduced)

    if not frames:
        raise FileNotFoundError(f"No Premier League E0.csv files found in {root}")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.dropna(subset=["Date", "FTR", "HomeTeam", "AwayTeam"])
    matches = matches.sort_values(["date", "season"], kind="stable").reset_index(
        drop=True
    )
    return matches


def build_pre_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    team_history: dict[str, list[dict[str, float]]] = {}
    feature_rows: list[dict[str, float]] = []

    for _, match in matches.iterrows():
        home = str(match["HomeTeam"])
        away = str(match["AwayTeam"])
        home_history = team_history.get(home, [])
        away_history = team_history.get(away, [])

        def average(history: list[dict[str, float]], field: str) -> float:
            recent = history[-5:]
            return float(np.mean([row[field] for row in recent])) if recent else np.nan

        feature_rows.append(
            {
                "home_goals_5": average(home_history, "goals_for"),
                "home_goals_against_5": average(home_history, "goals_against"),
                "home_points_5": average(home_history, "points"),
                "away_goals_5": average(away_history, "goals_for"),
                "away_goals_against_5": average(away_history, "goals_against"),
                "away_points_5": average(away_history, "points"),
                "B365H": float(match["B365H"]),
                "B365D": float(match["B365D"]),
                "B365A": float(match["B365A"]),
            }
        )

        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])
        team_history.setdefault(home, []).append(
            {
                "goals_for": home_goals,
                "goals_against": away_goals,
                "points": 3.0
                if home_goals > away_goals
                else 1.0
                if home_goals == away_goals
                else 0.0,
            }
        )
        team_history.setdefault(away, []).append(
            {
                "goals_for": away_goals,
                "goals_against": home_goals,
                "points": 3.0
                if away_goals > home_goals
                else 1.0
                if away_goals == home_goals
                else 0.0,
            }
        )

    return pd.DataFrame(feature_rows)


def train_model(
    matches: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[GaussianNB, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    complete = features.notna().all(axis=1)
    train_mask = matches["season"].isin(TRAIN_SEASONS) & complete
    test_mask = matches["season"].isin(TEST_SEASONS) & complete

    target = (matches["FTR"] == "H").astype(int)
    x_train = features.loc[train_mask].to_numpy(dtype=float)
    x_test = features.loc[test_mask].to_numpy(dtype=float)
    y_train = target.loc[train_mask].to_numpy(dtype=int)
    y_test = target.loc[test_mask]

    model = GaussianNB()
    model.fit(x_train, y_train)
    return model, x_test, y_test.to_numpy(dtype=int), y_train, y_test


def evaluate_model(
    data_dir: str | Path = "data",
) -> tuple[EvaluationReport, str]:
    matches = load_matches(data_dir)
    features = build_pre_match_features(matches)
    model, x_test, y_test, y_train, _ = train_model(matches, features)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["не победа хозяев", "победа хозяев"],
        zero_division=0,
        output_dict=True,
    )
    evaluation: EvaluationReport = {
        "accuracy": float(accuracy),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "classification_report": report,
    }
    text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["не победа хозяев", "победа хозяев"],
        zero_division=0,
    )
    return evaluation, text
