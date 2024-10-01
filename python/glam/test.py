import pandas as pd
from glam import DefaultModelData, BinomialGlmAnalysis
from glam.src.clusterers.feature_clusterers.decision_tree_feature_clusterer import DecisionTreeFeatureClusterer

import duckdb

def main():
    DATALOC = "~/glam/src/glam/cancer.parquet"
    df = duckdb.sql(f"from read_parquet('{DATALOC}')").df()
    df["cv"] = [i % 5 for i in range(len(df))]

    # subdivide mean_radius into centiles (100 quantiles) so that I have
    # a lot of clusters
    df["mean_radius_quantiles"] = pd.qcut(df["mean_radius"], 100, labels=[
        f"Q{i+1}" for i in range(100)
    ]).astype("category")
    
    print(f"Columns:\n{df.columns.tolist()}")
    data = DefaultModelData(df, y="target", cv="cv")

    print(f"Data: {data}")

    glm = BinomialGlmAnalysis(data)
    print(f"GLM (before first fit): {glm}")

    glm.add_feature("mean_radius")
    glm.add_feature("mean_texture")
    glm.add_feature("mean_perimeter")
    glm.fit()
    print(f"GLM (after first fit): {glm}")

    clusterer = DecisionTreeFeatureClusterer("mean_radius_quantiles", data)
    print(f"Clusterer (before fitting): {clusterer}")
 
    clusterer.fit(10)
    print(f"Clusterer (after fitting): {clusterer}")

    print("Mapping:")
    print(clusterer.mapping)

if __name__ == "__main__":
    main()
