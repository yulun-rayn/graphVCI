import numpy as np

from sklearn.metrics import r2_score

import torch

def evaluate_r2(model, dataset, batch_size=None, min_samples=30):
    """
    Measures different quality metrics about an DPI `model`, when
    tasked to translate some `genes_control` into each of the perturbation/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    mean_score, mean_score_de = [], []
    #var_score, var_score_de = [], []

    for pert_category in np.unique(dataset.cov_pert):
        # pert_category category contains: 'cov_pert' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        )[0]

        idx = np.where(dataset.cov_pert == pert_category)[0]

        # estimate metrics only for reasonably-sized perturbation/cell-type combos
        if len(idx) > min_samples:
            genes = dataset.genes[idx, :]
            num = genes.size(0)
            if batch_size is None:
                batch_size = num

            covars = [
                covar[idx][0].view(1, -1).repeat(num, 1).clone()
                for covar in dataset.covariates
            ]

            num_eval = 0
            yp = []
            while num_eval < num:
                end = min(num_eval+batch_size, num)
                out = model.predict(
                    genes[num_eval:end],
                    [covar[num_eval:end] for covar in covars]
                )
                yp.append(out.detach().cpu())

                num_eval += batch_size
            yp = torch.cat(yp, 0)
            yp_m = yp.mean(0)

            # true means
            yt = genes.numpy()
            yt_m = yt.mean(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]


def evaluate(autoencoder, datasets, batch_size=None):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training"].subset_condition(control=False),
                batch_size=batch_size
            ),
            "test": evaluate_r2(
                autoencoder, 
                datasets["test"].subset_condition(control=False),
                batch_size=batch_size
            ),
            "ood": evaluate_r2(
                autoencoder,
                datasets["ood"],
                batch_size=batch_size
            ),
            "optimal for perturbations": 1 / datasets["test"].num_perturbations
            if datasets["test"].num_perturbations > 0
            else None,
        }
    autoencoder.train()
    return evaluation_stats
