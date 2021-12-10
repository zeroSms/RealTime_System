import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, RFE, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class Wrapper_Method:
    def __init__(self, clf, X, y, make_file):
        self.clf = clf
        self.X, self.y = X, y
        self.make_file = make_file

    def RFE_CV(self):
        # 特徴量削減
        min_features_select = 10
        selector = RFECV(self.clf, min_features_to_select=min_features_select, cv=10)
        X_new = pd.DataFrame(selector.fit_transform(self.X, self.y),
                             columns=self.X.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=self.X.columns.values, columns=['False: dropped'])
        result['ranking'] = selector.ranking_
        result.to_csv(self.make_file + '\\feature_rank.csv')

        # Plot number of features VS. cross-validation scores
        fig = plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(min_features_select,
                       len(selector.grid_scores_) + min_features_select),
                 selector.grid_scores_)
        fig.savefig(self.make_file + '\\features_score.png')
        return X_new


class Embedded_Method:
    def __init__(self, clf, X, y, make_file):
        self.clf = clf
        self.X, self.y = X, y
        self.make_file = make_file

    def SFM(self):
        # 特徴量削減
        selector = SelectFromModel(self.clf, threshold=0.001709)  # 閾値以上の特徴量を選択
        X_new = pd.DataFrame(selector.fit_transform(self.X, self.y),
                             columns=self.X.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=self.X.columns.values, columns=['False: dropped'])
        result['featureImportances'] = selector.estimator_.feature_importances_
        result.to_csv(self.make_file + '\\feature_rank.csv')

        return X_new
