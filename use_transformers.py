import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from setup_utils import seed_everything

os.environ["WANDB_DISABLED"] = "true"

SEED = 0


def tokenize_and_encode(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def encode_dataset(dataset):
    cols = dataset.column_names
    cols.remove("labels")
    ds_enc = dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)
    ds_enc.set_format("torch")
    ds_enc = ds_enc.map(
        lambda x: {"float_labels": x["labels"].to(torch.float)},
        remove_columns=["labels"],
    ).rename_column("float_labels", "labels")
    return ds_enc


def train_transformer(train_ds, test_ds, transformer, save_model):
    seed_everything(SEED)
    num_labels = len(train_ds["labels"][0])
    print("number of labels: {} \n".format(num_labels))

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(transformer, padding=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        transformer, num_labels=num_labels, problem_type="multi_label_classification"
    ).to("cuda")

    train_ds_enc = encode_dataset(train_ds)
    test_ds_enc = encode_dataset(test_ds)

    training_args = TrainingArguments(
        num_train_epochs=5,
        logging_steps=200,
        evaluation_strategy="epoch",
        output_dir="./{}_training_output".format(transformer),
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_enc,
        # We run separate evaluation in do_eval -- evaluation on train data here is not considered
        eval_dataset=train_ds_enc,
        tokenizer=tokenizer,
    )

    trainer.train()

    train_predictions = trainer.predict(train_ds_enc)
    test_predictions = trainer.predict(test_ds_enc)

    if save_model == "full":
        direct = "written_models_full/" + transformer + "/"
        if not os.path.exists(direct):
            os.makedirs(direct)
        model.save_pretrained(direct)
        tokenizer.save_pretrained(direct)

    elif save_model == "collapse":
        direct = "written_models_collapse/" + transformer + "/"
        if not os.path.exists(direct):
            os.makedirs(direct)
        model.save_pretrained(direct)
        tokenizer.save_pretrained(direct)

    return train_predictions, test_predictions


def zero_shot(train_ds, test_ds, transformer):
    seed_everything(SEED)

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(transformer, padding=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        transformer, problem_type="multi_label_classification"
    ).to("cuda")

    train_ds_enc = encode_dataset(train_ds)
    test_ds_enc = encode_dataset(test_ds)

    training_args = TrainingArguments(
        num_train_epochs=0,
        logging_steps=200,
        evaluation_strategy="epoch",
        output_dir="./{}_training_output".format(transformer),
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_enc,
        # We run separate evaluation in do_eval -- evaluation on train data here is not considered
        eval_dataset=train_ds_enc,
        tokenizer=tokenizer,
    )

    train_predictions = trainer.predict(train_ds_enc)
    test_predictions = trainer.predict(test_ds_enc)

    return train_predictions, test_predictions
