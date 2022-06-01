import numpy as np
import json
from sklearn import metrics
from scipy.special import expit


def write_eval_json(preds, MODE, split):
    metric_dict = {"mac": [], "ap": []}
    threshold = 0.50
    for pred in preds:

        if "transformer" in MODE:
            logits = pred.predictions
            labels = pred.label_ids
            model_outputs = expit(logits)
            outputs = np.zeros((model_outputs.shape))
            outputs[model_outputs >= threshold] = 1

        else:
            labels, outputs, logits = pred

        macro = (
            metrics.f1_score(labels, outputs, zero_division=1, average="macro") * 100
        )

        avg_pre = (
            metrics.average_precision_score(labels, logits, average='samples')*100
        )

        metric_dict["mac"].append(macro)
        metric_dict["ap"].append(avg_pre)


    json_dict = dict()
    for metric, lst in metric_dict.items():
        json_dict[metric + "_{}".format("mean")] = np.mean(lst)
        json_dict[metric + "_{}".format("std")] = np.std(lst)

    writefile = "output/" + MODE.lower() + "_" + split + "_results.json"
    with open(writefile, "a") as f:
        f.write(json.dumps(json_dict) + "\n")
