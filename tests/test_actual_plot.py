from pdpbox import info_plots, get_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

test_titanic = get_dataset.titanic()

titanic_data = test_titanic["data"]
titanic_features = ["Fare"]  # test_titanic["features"]
titanic_target = test_titanic["target"]
# titanic_model = test_titanic["xgb_model"]
titanic_model = LogisticRegression()
titanic_model.fit(titanic_data[titanic_features], titanic_data[titanic_target])
X = titanic_data[titanic_features]
probs = titanic_model.predict_proba(X)[:, -1]


def test_actual_plot_binary():

    fig, axes, summary_df = info_plots.actual_plot_with_probs(
        probs=probs,
        X=titanic_data[titanic_features],
        feature="Fare",
        feature_name="Fare",
    )
    print(summary_df)
    assert fig is not None

