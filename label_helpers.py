"""Vendored parsing helpers (copied from ../label_subfacts.py) so the app is
self-contained and can be deployed without the rest of the research monorepo."""

import ast

import pandas as pd


def parse_cell(cell):
    """Safely parse a stringified python literal cell -> python object."""
    if isinstance(cell, (list, dict)):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []


def norm_crime(name):
    """Normalize a crime name for exact matching."""
    return " ".join(str(name).split()).strip()
