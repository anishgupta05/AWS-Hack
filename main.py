"""
Entry point. Runs the correction loop with no hooks wired in (Person B
will add on_action and zero_enrichment_hook when their integrations are
ready). Prints the full iteration log and final summary to stdout.

Target accuracy: 0.87 — strictly above the best native-data result
(0.859 at iter 8 in benchmarking), so convergence is only possible via
Zero.xyz enrichment. See CLAUDE.md for benchmark details.
"""

import logging

from src.loop.agent import run

logging.basicConfig(
    level=logging.WARNING,   # suppress internal chatter; agent prints its own logs
    format="%(message)s",
    stream=__import__("sys").stdout,
)

if __name__ == "__main__":
    state = run(target_accuracy=0.87, max_iterations=12)
    print("\n" + state.summary())
