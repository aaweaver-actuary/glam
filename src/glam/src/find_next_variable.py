"""Find the next variable to add to the model using a forward selection strategy."""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from glam.src.model_analysis import BaseModelAnalysis

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("find_next_variable.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def find_next_variable(glm: BaseModelAnalysis) -> str:
    """Find the next variable to add to the model using a forward selection strategy."""
    current_features = glm.features
    logger.debug(f"Current features ({len(current_features)}): {current_features}")
    remaining_features = [
        f
        for f in glm.data.feature_names
        if (f not in current_features)
        and (f not in glm.unanalyzed)
        and (f not in ["fold", "hit_count", "quote_count"])
    ]
    logger.debug(
        f"Remaining features ({len(remaining_features)}): {remaining_features}"
    )

    def mean_std(values: list[float]) -> tuple[float, float]:
        return np.mean(values), np.std(values)

    def evaluate_feature(
        glm: BaseModelAnalysis, f: str
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return upper and lower bounds for accuracy and AUC."""
        gcopy = glm.copy()
        try:
            logger.debug(f"Beginning evaluation of feature {f}")
            gcopy.add_feature(f)
            gcopy.fit()

            accuracy = mean_std(list(gcopy.accuracy.values()))
            logger.debug("Accuracy complete")
            precision = mean_std(list(gcopy.precision.values()))
            logger.debug("Precision complete")
            recall = mean_std(list(gcopy.recall.values()))
            logger.debug("Recall complete")
            f1 = mean_std(list(gcopy.f1.values()))
            logger.debug("F1 complete")
            auc_roc = mean_std(list(gcopy.auc_roc.values()))
            logger.debug("AUC complete")

            # glm statistics
            aic = gcopy.get_model(9).aic
            deviance = gcopy.get_model(9).deviance

            return (
                (accuracy[0] - accuracy[1], accuracy[0] + accuracy[1]),
                (precision[0] - precision[1], precision[0] + precision[1]),
                (recall[0] - recall[1], recall[0] + recall[1]),
                (f1[0] - f1[1], f1[0] + f1[1]),
                (auc_roc[0] - auc_roc[1], auc_roc[0] + auc_roc[1]),
                aic,
                deviance,
            )
        except Exception as e:
            logger.error(f"Error evaluating feature {f}: {e}")
            return ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), 0, 0)

    results = [evaluate_feature(glm, f) for f in tqdm(remaining_features)]

    output_df = (
        pd.DataFrame(
            {
                "feature": remaining_features,
                "accuracy_lower_bound": [r[0][0] for r in results],
                "precision_lower_bound": [r[1][0] for r in results],
                "recall_lower_bound": [r[2][0] for r in results],
                "f1_lower_bound": [r[3][0] for r in results],
                "auc_lower_bound": [r[4][0] for r in results],
            }
        )
        .assign(aic=[r[5] for r in results], deviance=[r[6] for r in results])
        .sort_values(by=["aic"], ascending=True)
    )

    output_df.to_csv("results.csv")

    return output_df["feature"].iloc[0]
