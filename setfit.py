import pandas as pd
import numpy as np
import torch
import os
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from setup_utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def sentence_pairs_generation(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative
    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSentence = sentences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))

    # return a 2-tuple of our image pairs and labels
    return pairs


def SetFit(train_df, sbert=None):
    seed_everything(SEED)
    st_model = "paraphrase-mpnet-base-v2"
    num_training = 32
    num_itr = 5
    text_col = train_df.columns.values[0]
    category_col = train_df.columns.values[1]

    examples = []
    for label in train_df["labels"].unique():
        subset = train_df[train_df["labels"] == label]
        if len(subset) > num_training:
            examples.append(subset.sample(num_training))
        else:
            examples.append(subset)
    train_df_sample = pd.concat(examples)
    x_train = train_df_sample[text_col].values.tolist()
    y_train = train_df_sample[category_col].values.tolist()

    train_examples = []
    for x in range(num_itr):
        train_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), train_examples
        )
    if sbert == None:
        model = SentenceTransformer(st_model).to(device)
    else:
        model = sbert

    # S-BERT adaptation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=10,
        show_progress_bar=True,
    )

    return model
