"""Test the GLAM package with the cancer dataset."""

import pandas as pd
from glam import DefaultModelData, BinomialGlmAnalysis
from glam.src.clusterers.feature_clusterers.decision_tree_feature_clusterer import ( # type: ignore
    DecisionTreeFeatureClusterer,
)

import duckdb
import logging


def main() -> None:
    """Test the GLAM package with the cancer dataset."""
    DATALOC = "~/glam/src/glam/cancer.parquet"
    df = duckdb.sql(f"from read_parquet('{DATALOC}')").df()
    df["cv"] = [i % 5 for i in range(len(df))]

    # subdivide mean_radius into centiles (100 quantiles) so that I have
    # a lot of clusters
    df["mean_radius_quantiles"] = pd.qcut(
        df["mean_radius"], 100, labels=[f"Q{i+1}" for i in range(100)]
    ).astype("category")

    logging.info(f"Columns:\n{df.columns.tolist()}")
    data = DefaultModelData(df, y="target", cv="cv")

    logging.info(f"Data: {data}")

    glm = BinomialGlmAnalysis(data)
    logging.info(f"GLM (before first fit): {glm}")

    glm.add_feature("mean_radius")
    glm.add_feature("mean_texture")
    glm.add_feature("mean_perimeter")
    glm.fit()
    logging.info(f"GLM (after first fit): {glm}")

    clusterer = DecisionTreeFeatureClusterer("mean_radius_quantiles", data)
    logging.info(f"Clusterer (before fitting): {clusterer}")

    clusterer.fit(10)
    logging.info(f"Clusterer (after fitting): {clusterer}")

    logging.info("Mapping:")
    logging.info(clusterer.mapping)


if __name__ == "__main__":
    main()
