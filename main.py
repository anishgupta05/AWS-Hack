"""
Entry point. Runs the correction loop with no hooks wired in (Person B
will add on_action and zero_enrichment_hook when their integrations are
ready). Prints the final summary to stdout.
"""

from src.loop.agent import run


if __name__ == "__main__":
    state = run(target_accuracy=0.833, max_iterations=6)
    print("\n" + state.summary())
