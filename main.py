import pandas as pd
import os
from datasets import Dataset
from setup_utils import label_fix, seed_everything, doKfold
from use_transformers import train_transformer, zero_shot
from all_vs_rest import lr_all_vs_rest
from sentence_transformers import SentenceTransformer
from setfit import SetFit
from do_eval import write_eval_json


SEED = 0


def main(meta_single, meta_multi, study2_single, study2_multi, MODE):
    print("working on mode {}".format(MODE))
    transformer = "roberta-base"
    train_preds, test_preds = [], []
    # pretrain model on specific level data
    if MODE in ["transformer_pretrain_full", "transformer_pretrain_collapse"]:
        train_ds = Dataset.from_pandas(meta_multi)
        test_ds = Dataset.from_pandas(meta_multi)

        if MODE == "transformer_pretrain_full":
            train_predictions, test_predictions = train_transformer(
                train_ds, test_ds, transformer, save_model="full"
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

        elif MODE == "transformer_pretrain_collapse":
            train_predictions, test_predictions = train_transformer(
                train_ds, test_ds, transformer, save_model="collapse"
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

    # use model primed on specific level data to perform zero shot classification on factchecks
    elif "transformer_zero" in MODE:
        train_ds = Dataset.from_pandas(meta_multi)
        test_ds = Dataset.from_pandas(study2_multi)
        if MODE == "transformer_zero_collapse":
            transformer = "written_models_collapse/" + transformer
            train_predictions, test_predictions = zero_shot(
                train_ds, test_ds, transformer
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

        elif MODE == "transformer_zero_full":
            transformer = "written_models_full/" + transformer
            train_predictions, test_predictions = zero_shot(
                train_ds, test_ds, transformer
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

    # perform standard finetuning and two-step finetuning with roberta-base
    elif "transformer_baseline" in MODE or "use_ft_transformer" in MODE:
        train_dfs, test_dfs = doKfold(study2_single, study2_multi)
        for idx, train_df in enumerate(train_dfs):
            transformer = "roberta-base"
            test_df = test_dfs[idx]
            train_df.drop("sing_labels", axis=1, inplace=True)
            test_df.drop("sing_labels", axis=1, inplace=True)
            train_ds = Dataset.from_pandas(train_df)
            train_ds = train_ds.rename_column("multi_labels", "labels")
            test_ds = Dataset.from_pandas(test_df)
            test_ds = test_ds.rename_column("multi_labels", "labels")

            if MODE == "use_ft_transformer_full":
                transformer = "written_models_full/" + transformer
                train_predictions, test_predictions = train_transformer(
                    train_ds, test_ds, transformer, save_model=False
                )
                train_preds.append(train_predictions)
                test_preds.append(test_predictions)

            elif MODE == "use_ft_transformer_collapse":
                transformer = "written_models_collapse/" + transformer
                train_predictions, test_predictions = train_transformer(
                    train_ds, test_ds, transformer, save_model=False
                )
                train_preds.append(train_predictions)
                test_preds.append(test_predictions)

            elif MODE in ["transformer_baseline_full", "transformer_baseline_collapse"]:
                train_predictions, test_predictions = train_transformer(
                    train_ds, test_ds, transformer, save_model=False
                )
                train_preds.append(train_predictions)
                test_preds.append(test_predictions)

    # Sentence Transformer baseline
    elif "st_baseline" in MODE:
        embed_model = SentenceTransformer("paraphrase-mpnet-base-v2")
        train_dfs, test_dfs = doKfold(study2_single, study2_multi)
        for idx, train_df in enumerate(train_dfs):
            test_df = test_dfs[idx]
            train_predictions, test_predictions = lr_all_vs_rest(
                embed_model, train_df, test_df
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

    # standard fine-tuning with SetFit
    elif "setfit_baseline" in MODE:
        train_dfs, test_dfs = doKfold(study2_single, study2_multi)
        for idx, train_df in enumerate(train_dfs):
            test_df = test_dfs[idx]
            sf_train_df = train_df.drop("multi_labels", axis=1)
            sf_train_df = sf_train_df.rename(columns={"sing_labels": "labels"})
            embed_model = SetFit(sf_train_df)
            train_predictions, test_predictions = lr_all_vs_rest(
                embed_model, train_df, test_df
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

    # two-step fine-tuning with SetFit
    elif "setfit_pretrain" in MODE:
        sbert = SetFit(meta_single)
        train_dfs, test_dfs = doKfold(study2_single, study2_multi)
        for idx, train_df in enumerate(train_dfs):
            test_df = test_dfs[idx]
            sf_train_df = train_df.drop("multi_labels", axis=1)
            sf_train_df = sf_train_df.rename(columns={"sing_labels": "labels"})
            embed_model = SetFit(sf_train_df, sbert)
            train_predictions, test_predictions = lr_all_vs_rest(
                embed_model, train_df, test_df
            )
            train_preds.append(train_predictions)
            test_preds.append(test_predictions)

    # zero-shot with SetFit
    elif "setfit_zero" in MODE:
        sbert = SetFit(meta_single)
        meta_multi = meta_multi.rename(columns={"labels": "multi_labels"})
        study2_multi = study2_multi.rename(columns={"labels": "multi_labels"})
        train_predictions, test_predictions = lr_all_vs_rest(
            sbert, meta_multi, study2_multi
        )
        train_preds.append(train_predictions)
        test_preds.append(test_predictions)

    write_eval_json(train_preds, MODE, "train")
    write_eval_json(test_preds, MODE, "test")


if __name__ == "__main__":

    # read in single label datasets with all attitude roots
    # single label data are needed to fine-tune the Setfit embedding model
    # these data are created by only considering examples with a single label in Study 1 or are explicitly given in Study 2 data
    meta_single_all = pd.read_csv("data/full_single_label/meta_single_label.csv")
    study2_single_all = pd.read_csv("data/full_single_label/study_2_single_label.csv")
    meta_single_all = meta_single_all.dropna()
    study2_single_all = study2_single_all.dropna()

    # read in single label datasets with collapased attitude roots
    meta_single_col = pd.read_csv("data/single_collapse/meta_single_label_collapse.csv")
    study2_single_col = pd.read_csv(
        "data/single_collapse/study2_single_label_collpase.csv"
    )
    meta_single_col = meta_single_col.dropna()
    study2_single_col = study2_single_col.dropna()

    # read in multilabel datasets with all attitude roots
    meta_multi_all = pd.read_csv("data/full_multilabel/meta_multi_full.csv")
    meta_multi_all["labels"] = meta_multi_all["labels"].apply(label_fix)
    study2_multi_all = pd.read_csv("data/full_multilabel/study2_full_all_labels.csv")
    study2_multi_all["labels"] = study2_multi_all["labels"].apply(label_fix)
    meta_multi_all = meta_multi_all.dropna()
    study2_multi_all = study2_multi_all.dropna()

    # read in multilabel datasets with collapsed attitude roots
    meta_multi_col = pd.read_csv("data/multi_collapse/multilabel_meta_collapsed.csv")
    meta_multi_col["labels"] = meta_multi_col["labels"].apply(label_fix)
    study2_multi_col = pd.read_csv("data/multi_collapse/study2_multi_collapsed.csv")
    study2_multi_col["labels"] = study2_multi_col["labels"].apply(label_fix)
    meta_multi_col = meta_multi_col.dropna()
    study2_multi_col = study2_multi_col.dropna()
    # pretrain = train models on study1 data
    # full = 11 attitude roots
    # collapse = 7 attitude roots
    # transformer_zero = use transformer for zero shot
    # transformer_baseline = standard fine-tuning
    # use ft = two-step finetuning with transformer pretrained on study 1 data
    # st_baseline = Sentence Transformers and logistic regressions baseline
    # setfit_baseline = setfit standard fine-tuning on study 2 fact checks
    # setfit_pretrain = two-step fine-tuning with setfit
    # setfit_zero = zero-shot on study two fact checks with setfit

    MODES = [
        "transformer_pretrain_full",
        "transformer_pretrain_collapse",
        "transformer_zero_full",
        "transformer_zero_collapse",
        "transformer_baseline_full",
        "transformer_baseline_collapse",
        "use_ft_transformer_full",
        "use_ft_transformer_collapse",
        "st_baseline_full",
        "st_baseline_collapse",
        "setfit_baseline_full",
        "setfit_baseline_collapse",
        "setfit_pretrain_full",
        "setfit_pretrain_collapse",
        "setfit_zero_full",
        "setfit_zero_collapse",
    ]

    print(len(MODES))
    print(len(set(MODES)))

    for mode in MODES:
        print(mode)
        if "_collapse" in mode:
            main(
                meta_single_col,
                meta_multi_col,
                study2_single_col,
                study2_multi_col,
                mode,
            )
        elif "_full" in mode:
            main(
                meta_single_all,
                meta_multi_all,
                study2_single_all,
                study2_multi_all,
                mode,
            )

    print("Job's done!")
