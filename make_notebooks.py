"""Build the platform notebooks for any day from the solution files, and test each code cell.

Every solution file starts with a docstring like
    "Day 19 / Session 1 - Lab Assignment 1: <title>"
and the platform notebook is named d<day>_s<session>_a<assignment>.ipynb.
A file may hold only code that the notebook needs above the  if __name__ == "__main__":  line.

Usage:   python make_notebooks.py 19 20 21        (days to build; default: every solution file found)
"""
import glob
import json
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "notebooks")
os.makedirs(out, exist_ok=True)
header = re.compile(r"Day (\d+) / Session (\d+) - Lab Assignment (\d+)")
wanted = set(int(a) for a in sys.argv[1:])

made = []
bad = []
for path in sorted(glob.glob(os.path.join(here, "[0-9]*lb[0-9]*.py"))):
    source = open(path).read()
    found = header.search(source)
    if not found:
        continue
    day, session, assignment = (int(g) for g in found.groups())
    if wanted and day not in wanted:
        continue
    code = source.split('\n\nif __name__ == "__main__":')[0].rstrip("\n") + "\n"
    name = f"d{day}_s{session}_a{assignment}.ipynb"
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": code.splitlines(True)}
    notebook = {"cells": [cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                              "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(out, name), "w") as f:
        json.dump(notebook, f, indent=1)
    made.append(name)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/tmp")
    if result.returncode:
        bad.append((name, result.stderr.strip().splitlines()[-1:]))

print(made)
print("code-cell failures:", bad)
