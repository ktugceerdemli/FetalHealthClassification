from utils import (StratifiedKFold, make_scorer, roc_auc_score, accuracy_score, 
                   f1_score, precision_score, recall_score, LogisticRegression)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_params = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 10.0, 100, 1000, 10000],
    'max_iter': [100, 200, 300, 400],
    'class_weight': [None, 'balanced'],
    'tol': [1e-3, 1e-4, 1e-5]
}

all_scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='weighted'),
    'roc_auc_score': make_scorer(lambda y_in, y_p_in: roc_auc_score(y_in, y_p_in, multi_class='ovr'), needs_proba=True)}

classifiers = [("LR", LogisticRegression(random_state=42), lr_params)]