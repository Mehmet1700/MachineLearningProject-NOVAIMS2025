# utils/feature_selection.py
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    SelectFromModel,
)
from sklearn.linear_model import LassoCV


def candidate_selectors(random_state: int = 42):
    """
    High-Performance Feature Selection Mode (Option C):
    - Fokus auf effiziente, robuste, moderne Methoden
    - kein RFE (zu langsam)
    - kein RandomForest (zu schwer)
    - LassoCV mit cv=3: guter Kompromiss aus Qualität & Speed
    """

    candidates = {}

    # 1) No Feature Selection (Baseline)
    candidates["none"] = "passthrough"

    # 2) Filter Selection (sehr effizient, gute baseline)
    candidates["filter_freg_k100"] = SelectKBest(
        score_func=f_regression,
        k=100,   # gute Default-Wahl — stabil und effizient
    )

    # 3) Embedded Selection mittels LassoCV (sehr stark für Regressionsprobleme)
    lasso = LassoCV(
        cv=3,  # effizienter als cv=5 / 10
        random_state=random_state,
        n_jobs=-1,
    )
    candidates["lasso_embedded"] = SelectFromModel(
        estimator=lasso,
        threshold="median",   # wählt ca. die Hälfte aller Features
    )

    return candidates
