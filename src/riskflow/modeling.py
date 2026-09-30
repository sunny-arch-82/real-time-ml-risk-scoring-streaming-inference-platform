import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
from .features import NUMERIC,CATEGORICAL

def make_model(kind,columns,seed):
    numeric=[column for column in columns if column in NUMERIC]
    categorical=[column for column in columns if column in CATEGORICAL]
    preprocess=ColumnTransformer([("numeric",StandardScaler(),numeric),("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)])
    if kind=="logistic":
        estimator=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=seed)
    elif kind=="random_forest":
        estimator=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=3,class_weight="balanced_subsample",random_state=seed,n_jobs=1)
    else:
        raise ValueError("Unknown model family")
    return Pipeline([("preprocess",preprocess),("model",estimator)])

def fit_model(kind,x,y,columns,config,tune=False):
    model=make_model(kind,columns,config.seed)
    if not tune:
        return model.fit(x,y),None
    cv=list(TimeSeriesSplit(n_splits=config.cv_splits,gap=config.cv_gap).split(x))
    for train,validation in cv:
        if len(np.unique(np.asarray(y)[train]))<2 or len(np.unique(np.asarray(y)[validation]))<2:
            raise ValueError("Temporal CV fold lacks both classes; enlarge dataset or reduce CV splits")
    space=config.tuning_space.get(kind)
    if not space:
        raise ValueError(f"No tuning space supplied for {kind}")
    search=RandomizedSearchCV(model,space,n_iter=config.search_iterations,scoring="average_precision",cv=cv,random_state=config.seed,n_jobs=1,error_score="raise",refit=True)
    search.fit(x,y)
    return search.best_estimator_,{"best_parameters":search.best_params_,"best_cv_average_precision":float(search.best_score_),"cv_splits":config.cv_splits,"gap_rows":config.cv_gap}
