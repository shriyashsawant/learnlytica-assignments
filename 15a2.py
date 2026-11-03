"""Day 15 / Session 2 - Assignment: A Reusable Model Bake-off on a New Dataset."""


# # Day 15 – A Reusable Model Bake-off on a New Dataset
# 
# Imbalanced binary classification: shared folds, one metric, four+ pipelines, GridSearchCV + RandomizedSearchCV, a single held-out test evaluation, a leakage demo and a saved model.


import numpy as np, pandas as pd, joblib
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 42

# Task 1: generate the data in-code (imbalanced via weights) and split off the stratified 20% test set FIRST
X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=4,
                           weights=[0.85, 0.15], flip_y=0.02, class_sep=1.0, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("train", X_train.shape, "test", X_test.shape)
print("class balance (train):", np.bincount(y_train) / len(y_train))


# ## Task 2 – shared folds and metric
# 
# The data is imbalanced (about 85% / 15%), so plain accuracy would reward always predicting the majority class. I use **f1_macro**: it averages the F1 of each class equally, so the minority class matters as much as the majority class. Every model is scored on the SAME `StratifiedKFold` object so the comparison is fair.


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = "f1_macro"


# ## Task 3 – candidate pipelines
# 
# Scale-sensitive models (logistic regression, SVC, kNN) get a `StandardScaler` inside the pipeline; tree ensembles do not need one.


candidates = {
    "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "svc":    Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=RANDOM_STATE))]),
    "knn":    Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "rf":     Pipeline([("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
    "gb":     Pipeline([("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
}
print(list(candidates))


# ## Task 4 – screen all candidates on the shared folds


rows = []
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
leaderboard = pd.DataFrame(rows).set_index("model").sort_values("cv_mean", ascending=False)
print(leaderboard)


# ## Task 5 – tune the top two models
# 
# The best model gets a small discrete `GridSearchCV`; the runner-up gets a `RandomizedSearchCV` with distributions and `n_iter`. Both reuse the same `cv` and metric.


grid_spaces = {
    "logreg": {"clf__C": [0.1, 1, 10]},
    "svc":    {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]},
    "knn":    {"clf__n_neighbors": [3, 5, 11, 21]},
    "rf":     {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
    "gb":     {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
}
rand_spaces = {
    "logreg": {"clf__C": loguniform(1e-2, 1e2)},
    "svc":    {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-3, 1e0)},
    "knn":    {"clf__n_neighbors": randint(3, 40)},
    "rf":     {"clf__n_estimators": randint(100, 300), "clf__max_depth": randint(3, 20)},
    "gb":     {"clf__n_estimators": randint(100, 250), "clf__learning_rate": uniform(0.02, 0.18)},
}
top1, top2 = leaderboard.index[0], leaderboard.index[1]

grid = GridSearchCV(clone(candidates[top1]), grid_spaces[top1], cv=cv, scoring=SCORING, n_jobs=-1, refit=True)
grid.fit(X_train, y_train)
rand = RandomizedSearchCV(clone(candidates[top2]), rand_spaces[top2], n_iter=15, cv=cv, scoring=SCORING,
                          n_jobs=-1, random_state=RANDOM_STATE, refit=True)
rand.fit(X_train, y_train)

print("GridSearchCV      ", top1, "best_params_:", grid.best_params_, "best_score_: %.4f" % grid.best_score_)
print("RandomizedSearchCV", top2, "best_params_:", rand.best_params_, "best_score_: %.4f" % rand.best_score_)


# ## Task 6 – pick the winner and evaluate once on the held-out test set


winner_name, winner = (top1, grid) if grid.best_score_ >= rand.best_score_ else (top2, rand)
best_model = winner.best_estimator_            # refit=True already refit on all of X_train
y_pred = best_model.predict(X_test)            # the ONLY use of the test set for scoring
test_f1 = f1_score(y_test, y_pred, average="macro")
print("winner:", winner_name, "| CV f1_macro: %.4f" % winner.best_score_, "| test f1_macro: %.4f" % test_f1)
print(classification_report(y_test, y_pred, digits=3))


# ## Task 7 – leakage demonstration
# 
# Scale-sensitive model (SVC). **Leaky**: the scaler is fit on all training data first, so each validation fold has already influenced the scaling. **Clean**: the scaler lives inside a `Pipeline`, so it is re-fit on the training part of every fold only.


from sklearn.model_selection import RepeatedStratifiedKFold
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)   # 50 folds -> less noisy comparison
scaler_outside = StandardScaler().fit(X_train)            # fit OUTSIDE cross-validation -> leakage
X_train_leaky = scaler_outside.transform(X_train)
leaked = cross_val_score(SVC(random_state=RANDOM_STATE), X_train_leaky, y_train, cv=rcv, scoring=SCORING, n_jobs=-1)
clean = cross_val_score(Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=RANDOM_STATE))]),
                        X_train, y_train, cv=rcv, scoring=SCORING, n_jobs=-1)
print("leaked CV f1_macro: %.4f | clean (Pipeline) CV f1_macro: %.4f | gap (leaked - clean): %+.4f | leaked >= clean: %s"
      % (leaked.mean(), clean.mean(), leaked.mean() - clean.mean(), leaked.mean() >= clean.mean()))
# Comment: in the leaky version the scaler has already seen every validation fold, so the fold estimate is not a
# true out-of-sample estimate and can only be optimistic or neutral, never more honest. With plain standardisation
# the effect is tiny (compare the gap with the fold std), but the habit is wrong and with target-aware steps such as
# feature selection or imputation the optimism is large. Fix: keep every fitted preprocessing step inside the Pipeline.


# ## Task 8 – persist, reload and confirm the same test score


joblib.dump(best_model, "best_model.joblib")
reloaded = joblib.load("best_model.joblib")
reloaded_f1 = f1_score(y_test, reloaded.predict(X_test), average="macro")
print("original test f1_macro: %.4f | reloaded test f1_macro: %.4f" % (test_f1, reloaded_f1))
assert np.isclose(test_f1, reloaded_f1)
print("reloaded model reproduces the held-out score")


# ## Conclusion
# 
# Support Vector Classifier won the bake-off. In the untuned screen gradient boosting led (CV f1_macro 0.7288 +/- 0.0228), but SVC (0.7256 +/- 0.0345) and random forest (0.7237 +/- 0.0157) were within one std of it, so the untuned ranking was effectively a three-way tie. Tuning changed the ranking: RandomizedSearchCV lifted the SVC to a best_score_ of 0.7728 (C about 15.70, gamma about 0.0625) while GridSearchCV on gradient boosting reached 0.7534, so the SVC moved from second to first, although the 0.0194 gap to tuned gradient boosting is smaller than the roughly 0.02-0.03 fold std, i.e. a close finisher rather than a decisive win. On the held-out test set the SVC scored f1_macro 0.7809 (minority-class F1 0.628), slightly above its CV score of 0.7728; the 0.0081 difference is well inside the fold std, so the CV estimate looks honest rather than optimistic. The reloaded joblib model reproduced exactly 0.7809. The leakage demo (scaler fit outside CV vs inside a Pipeline) moved the estimate by only a few thousandths, far less than the fold std, so scaling leakage is small here but the Pipeline habit is still the safe one.
