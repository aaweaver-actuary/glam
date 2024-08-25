# """Plot exploratory data analysis (EDA) plots for a given dataset."""

# import plotly.graph_objects as go
# import pandas as pd
# import polars as pl
# import polars.selectors as cs
# from glam.src.data.base_model_data import BaseModelData
# from glam.src.fitted_model.base_fitted_model import BaseFittedModel
# from glam.src.model_list.base_model_list import BaseModelList

# class EdaPlotter:

#     def __init__(self, data: BaseModelData, models: BaseModelList, feature: str, measure_name: str, exposure: str | None = None):
#         self._data = data
#         self._models = models
#         self._feature = feature
#         self._measure_name = measure_name
#         self._exposure = exposure

#     @property
#     def models(self):
#         """The models property."""
#         return self._models

#     @property
#     def X(self):
#         """The X property."""
#         return self._data.X[self._feature]

#     @property
#     def y(self):
#         """The y property."""
#         return self._data.y

#     @property
#     def exposure(self):
#         """The exposure property."""
#         return self._data.X[self._exposure] if self._exposure else pd.Series([1] * len(self._data.X))

#     @property
#     def df(self):
#         """The df property."""
#         df = pd.concat([self.X, self.y, self.exposure], axis=1)
#         df.columns = [self._feature, self._data.y.name, "exposure"]
#         return df

#     @property
#     def is_categorical(self):
#         """The is_categorical property."""
#         categorical_cols = pl.from_pandas(self.df).select([cs.string(), cs.categorical()])
#         return self._feature in categorical_cols.columns

#     def _assign_quintiles(self):
#         """Assign quintiles to the data."""
#         self.df["Quintile"] = pd.qcut(self.df[self._feature], q=5, labels=[f"Q{i}" for i in range(1, 6)])

#     def _prep_categorical_data(self) -> pd.DataFrame:
#         """Prepare the data for a categorical feature by grouping and aggregating, then sorting by the measure."""
#         df = self.df.groupby([self._feature]).sum().reset_index()
#         df[self._measure_name] = df[self._data.y.name] / df["exposure"]

#         return df[[self._feature, self._measure_name]].sort_values(by=self._measure_name, ascending=False)

#     def _prep_continuous_data(self) -> pd.DataFrame:
#         """Prepare the data for a continuous feature by binning into quintiles, calculating the average measure, and sorting by the bin label."""
#         self._assign_quintiles()
#         df = self.df.copy()
#         df = df.groupby(["Quintile"]).sum().reset_index()
#         df[self._measure_name] = df[self._data.y.name] / df["exposure"]

#         return df[["Quintile", self._measure_name]].sort_values(by="Quintile")

#     def _plot_feature_data(self, fig: go.Figure | None = None) -> go.Figure:
#         """Plot the feature data."""
#         df = self._prep_categorical_data() if self.is_categorical else self._prep_continuous_data()

#         fig = go.Figure() if fig is None else fig
#         fig.add_trace(
#             go.Scatter(
#                 x=df[self._feature],
#                 y=df[self._measure_name],
#                 mode="lines+markers",
#                 line=dict(color="blue", width=2),
#                 marker=dict(
#                     size=10,
#                     color="blue",
#                     opacity=0.5,
#                     line=dict(color="black", width=1),
#                 ),
#                 name=self._feature,
#             )
#         )

#         return fig

#     def _plot_model_data(self, model: BaseFittedModel, fig: go.Figure | None = None) -> go.Figure:
#         """Plot the model data."""
#         fig = go.Figure() if fig is None else fig
#         model_data = model.yhat_proba(self._data.X)
#         feature_quintiles = pd.qcut(self.X, q=5, labels=[f"Q{i}" for i in range(1, 6)])
#         df = pd.DataFrame(
#             {
#                 self._feature: feature_quintiles,
#                 f"Model {}": model_data,
#             }
#         ).groupby([self._feature]).mean().reset_index()
