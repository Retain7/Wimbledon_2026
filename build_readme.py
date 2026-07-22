"""
build_readme.py - splice generated result blocks into README.md.

Every number in the README comes from a script, not from a copy-paste. Each
results/*.md file is written into the matching marker pair in README.md:

    <!-- BEGIN:model_report -->
    ...replaced...
    <!-- END:model_report -->

Run after regenerating results:
    python wimbledon_rf.py
    python odds_backtest.py --years 2018 2019 2021 2022 2023 2024 2025
    python postmortem.py
    python build_readme.py

--check exits non-zero if the README is out of date with results/, which is
what you want in CI or as a pre-push habit.
"""

import os
import re
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(script_dir, "README.md")
RESULTS = os.path.join(script_dir, "results")

BLOCKS = {
    "backtest_report": "backtest_report.md",
    "model_report": "model_report.md",
    "postmortem_report": "postmortem_report.md",
}

PLACEHOLDER = ("_Not generated yet. Run the commands in "
               "[Reproducing](#reproducing), then `python build_readme.py`._")


def splice(text, name, body):
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise KeyError(f"No {begin} ... {end} marker pair in README.md")
    return pattern.sub(f"{begin}\n{body.strip()}\n{end}", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if README is out of date, write nothing")
    args = parser.parse_args()

    with open(README) as f:
        original = f.read()

    text = original
    missing = []
    for name, filename in BLOCKS.items():
        path = os.path.join(RESULTS, filename)
        if os.path.exists(path):
            with open(path) as f:
                text = splice(text, name, f.read())
        else:
            missing.append(filename)
            text = splice(text, name, PLACEHOLDER)

    if args.check:
        if text != original:
            print("README.md is out of date with results/. Run build_readme.py.")
            return 1
        print("README.md is up to date.")
        return 0

    with open(README, "w") as f:
        f.write(text)

    for name in BLOCKS:
        print(f"  spliced {name}")
    if missing:
        print(f"\nMissing (placeholder inserted): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())