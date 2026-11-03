"""Build the Day 13 notebooks with the names the platform questions require, and test each lab code cell."""
import json
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "notebooks")
os.makedirs(out, exist_ok=True)


def write(name, code):
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": code.rstrip("\n").splitlines(True)}
    nb = {"cells": [cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(out, name), "w") as f:
        json.dump(nb, f, indent=1)


bad = []
for s in (1, 2, 3):
    for k, a in ((2 * s - 1, 1), (2 * s, 2)):
        code = open(os.path.join(here, f"13lb{k}.py")).read().split('\n\nif __name__ == "__main__":')[0]
        name = f"d13_s{s}_a{a}.ipynb"
        write(name, code)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/tmp")
        if r.returncode:
            bad.append((name, r.stderr.strip().splitlines()[-1:]))
    write(f"day13_s0{s}.ipynb", open(os.path.join(here, f"13a{s}.py")).read())

print(sorted(os.listdir(out)))
print("lab code failures:", bad)
