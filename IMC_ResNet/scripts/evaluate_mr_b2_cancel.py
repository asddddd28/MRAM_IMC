"""Replay b2 cancellation on MR3/MR2 only; keep forwarding unchanged."""
from evaluate_mr_positive_forward import main

if __name__ == "__main__":
    main(b2=True)
