"""
End-to-End Machine Learning Project — California Housing Prices
Vereisten: scikit-learn >= 1.0.1, scipy, pandas, numpy, joblib
"""

# ─────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import randint

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


# ─────────────────────────────────────────────
# 2. Data laden (alleen downloaden indien nodig)
# ─────────────────────────────────────────────
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()


# ─────────────────────────────────────────────
# 3. Train/test split (gestratificeerd op inkomen)
# ─────────────────────────────────────────────
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# ─────────────────────────────────────────────
# 4. Custom transformers
# ─────────────────────────────────────────────
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# ─────────────────────────────────────────────
# 5. Preprocessing pipeline
# ─────────────────────────────────────────────
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("bedrooms",        ratio_pipeline(),  ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(),  ["total_rooms", "households"]),
    ("people_per_house",ratio_pipeline(),  ["population", "households"]),
    ("log",             log_pipeline,      ["total_bedrooms", "total_rooms",
                                            "population", "households", "median_income"]),
    ("geo",             cluster_simil,     ["latitude", "longitude"]),
    ("cat",             cat_pipeline,      make_column_selector(dtype_include=object)),
], remainder=default_num_pipeline)  # houdt housing_median_age over


# ─────────────────────────────────────────────
# 6. Volledige pipeline (preprocessing + model)
# ─────────────────────────────────────────────
full_pipeline = Pipeline([
    ("preprocessing",  preprocessing),
    ("random_forest",  RandomForestRegressor(random_state=42)),
])


# ─────────────────────────────────────────────
# 7. Hyperparameter tuning met RandomizedSearchCV
# ─────────────────────────────────────────────
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features":    randint(low=2, high=20),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42
)

rnd_search.fit(housing, housing_labels)
print("Beste parameters:", rnd_search.best_params_)


# ─────────────────────────────────────────────
# 8. Evaluatie op de testset
# ─────────────────────────────────────────────
final_model = rnd_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test  = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(f"Test RMSE: {final_rmse:,.0f}")


# ─────────────────────────────────────────────
# 9. Pipeline opslaan (inclusief preprocessing)
# ─────────────────────────────────────────────
Path("model").mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, "model/california_housing_model.pkl")
print("Model opgeslagen als model/california_housing_model.pkl")