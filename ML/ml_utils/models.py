from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


_HYPERPARAM_DISTS = {
    "SVC": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale"],
        "model__probability": [True],
    },
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
        "model__max_iter": [2000],
        "model__multi_class": ["multinomial"],
    },
    "RandomForest": {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 0.7],
    },
    "ExtraTrees": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.7, 1.0],
        "model__min_samples_leaf": [5, 10],
        "model__max_features": ["sqrt", None],
    },
    "HistGradientBoosting": {
        "model__max_iter": [300, 600],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [None, 3, 6],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__min_samples_leaf": [10, 20, 30],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__early_stopping": [True],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "DecisionTree": {
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GaussianNB": {},
    "MLP": {
        "model__hidden_layer_sizes": [(100,), (64, 64)],
        "model__alpha": [1e-4, 1e-3],
        "model__max_iter": [800],
    },
    "AdaBoost": {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.05, 0.1, 0.3],
    },
    "XGBoost": {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.03, 0.1],
        "model__n_jobs": [1],
        "model__verbosity": [0],
        "model__max_depth": [3, 5],
        "model__subsample": [0.7, 1.0],
        "model__colsample_bytree": [0.7, 1.0],
        "model__reg_lambda": [0.0, 1.0, 5.0],
        "model__reg_alpha": [0.0, 0.5],
        "model__tree_method": ["hist"],
        "model__eval_metric": ["logloss"],
    },
    "CatBoost": {
        "model__iterations": [400, 800],
        "model__learning_rate": [0.03, 0.1],
        "model__depth": [4, 6],
        "model__l2_leaf_reg": [3.0, 6.0, 10.0],
        "model__loss_function": ["MultiClass"],
        "model__od_type": ["Iter"],
        "model__od_wait": [50],
        "model__grow_policy": ["SymmetricTree"],
        "model__verbose": [False],
    },
    "QDA": {
        "model__reg_param": [0.05, 0.1, 0.2, 0.5],
        "model__store_covariance": [False],
    },
    "LDA": [
        {"model__solver": ["svd"]},
        {"model__solver": ["lsqr"], "model__shrinkage": [None, "auto"]},
    ],
}


def get_classifiers(seed=42):
    return [
        ("LDA", LinearDiscriminantAnalysis()),
        ("SVC", SVC(probability=True, class_weight="balanced", random_state=seed)),
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        (
            "RandomForest",
            RandomForestClassifier(
                random_state=seed, class_weight="balanced_subsample"
            ),
        ),
        ("ExtraTrees", ExtraTreesClassifier(random_state=seed)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=seed)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=seed)),
        ("KNN", KNeighborsClassifier()),
        ("DecisionTree", DecisionTreeClassifier(random_state=seed)),
        ("GaussianNB", GaussianNB()),
        ("MLP", MLPClassifier(max_iter=800, random_state=seed)),
        ("AdaBoost", AdaBoostClassifier(random_state=seed)),
        (
            "XGBoost",
            XGBClassifier(
                eval_metric="logloss",
                verbosity=0,
                random_state=seed,
                tree_method="hist",
                n_jobs=1,
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(verbose=0, random_seed=seed, loss_function="Logloss",thread_count=1),
        ),
        ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.01)),
    ]


def iter_model_specs(seed=42):
    name2est = dict(get_classifiers(seed=seed))

    for name, est in name2est.items():
        grid = _HYPERPARAM_DISTS.get(name, {})
        yield name, est, grid
