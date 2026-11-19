from .bayesian import (
    evaluate_model,
)


def main() -> None:
    model, report = evaluate_model()
    print(report)
    print(f"Accuracy: {model['accuracy']:.3f}")
