from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from setup_utils import seed_everything

SEED = 0


def lr_all_vs_rest(embed_model, train_df, test_df):
    seed_everything(SEED)
    lr = LogisticRegression(random_state=SEED)
    ovr_classifier = OneVsRestClassifier(lr, n_jobs=-1)

    X_train = embed_model.encode(train_df["text"].to_list())
    X_test = embed_model.encode(test_df["text"].to_list())

    y_train = train_df["multi_labels"].to_list()
    y_test = test_df["multi_labels"].to_list()

    ovr_classifier.fit(X_train, y_train)

    pred_train = ovr_classifier.predict(X_train)
    logit_train = ovr_classifier.predict_proba(X_train)

    train_predictions = (y_train, pred_train, logit_train)

    pred_test = ovr_classifier.predict(X_test)
    logit_test = ovr_classifier.predict_proba(X_test)

    test_predictions = (y_test, pred_test, logit_test)

    return train_predictions, test_predictions
