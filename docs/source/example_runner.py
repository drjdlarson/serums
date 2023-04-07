import sys
import os


def _run_distribution_overbounder():
    sys.path.insert(0, os.path.abspath("./example_scripts/distribution_overbounder"))
    import fusion_gaussian as f_gauss
    import fusion_gps as f_gps

    f_gauss.run()
    f_gps.run()

    sys.path.pop(0)


def run_examples():
    _run_distribution_overbounder()
