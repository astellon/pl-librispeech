from typing import Tuple

import torch
import torch.nn.functional as F
import torchmetrics


def read_trials(path: str) -> Tuple[list[int], list[Tuple[str, str]]]:
    labels = []
    trials = []

    with open(path, 'r') as f:
        for line in f:
            label, enrollment, test = line.rstrip().split(' ')
            labels.append(int(label))
            trials.append((enrollment, test))

    return torch.tensor(labels), trials


def compute_scores(embeddings: dict[str, torch.Tensor], trials: list[Tuple[str, str]]):
    scores = []

    for (enrollment, test) in trials:
        e_e = embeddings[enrollment]
        e_t = embeddings[test]

        score = F.cosine_similarity(e_e.view(1, -1), e_t.view(1, -1)).item()

        scores.append(score)

    return torch.tensor(scores)


def compute_eer(scores, labels) -> Tuple[float, float]:
    fpr, tpr, thresholds = torchmetrics.functional.roc(scores, labels, pos_label=1)
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    opt = torch.argmin(torch.abs((fnr - fpr)))
    eer = max(fpr[opt], fnr[opt])

    return eer, thresholds[opt]


if __name__ == '__main__':
    labels, trials = read_trials('data/materials/dev-trials.txt')
    scores = torch.rand(len(labels))
    eer, threshold = compute_eer(scores, labels)
    print(eer)
