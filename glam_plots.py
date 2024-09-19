import pandas as pd
import numpy as np
import polars as pl

def categorical_plot(model, data, test_feature="business_category"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model2 = model.copy()
    model2.add_feature(test_feature)
    model2.fit()

    df = pd.concat([
            model.y,
            model.cv,    
            model.X,
        ], axis=1)
    df = df.loc[df['fold'] >= 10]
    df[test_feature] = data.df[test_feature]
    df['current_model'] = model.yhat_proba(df.drop(columns=[model.y.name, model.cv.name, test_feature]))
    df['test_model'] = model2.yhat_proba(df.drop(columns=[model.y.name, model.cv.name]))

    count_by_level = df.groupby(test_feature).count()[['hit_count']].reset_index().rename(columns={'hit_count': 'count'})

    ave_by_level = (
        df.groupby(test_feature)
        .mean()[['hit_count', 'current_model', 'test_model']]
        .reset_index()
        .set_index(test_feature)
        .join(
            count_by_level.set_index(test_feature),
            on=test_feature
        ).reset_index()
        .sort_values(test_feature, ascending=True)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=ave_by_level[test_feature],
            y=ave_by_level['count'],
            name='Count',
            marker=dict(
                color='gray',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            # do not show legend for this trace
            showlegend=False
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level['hit_count'],
            mode='lines+markers',
            name='Actual',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            line=dict(
                color='blue',
                width=2
            ),

        ),
            secondary_y=False
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level['current_model'],
            mode='lines+markers',
            name='Current Model',
            marker=dict(
                size=10,
                color='red',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            line=dict(
                color='red',
                width=1.5,
                dash='dashdot'
            ),
        ),
            secondary_y=False
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level[test_feature],
            y=ave_by_level['test_model'],
            mode='lines+markers',
            name='Test Model',
            marker=dict(
                size=7,
                color='green',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            line=dict(
                color='green',
                width=1.5,
                dash='dashdot'
            ),
        ),
            secondary_y=False
    )

    fig.update_layout(
        title='Actual vs. Model Predictions',
        xaxis_title=test_feature,
        yaxis_title='Hit Ratio',
        showlegend=True
    )

    fig.update_yaxes(title_text="Number of Quotes", secondary_y=True)

    return fig

def numeric_plot(model, data, test_feature, n_bins=5):
    """Bin the numeric feature into n_bins and plot the actual hit ratio and model predictions"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model2 = model.copy()
    model2.add_feature(test_feature)
    model2.fit()

    df = pd.concat([
            model.y,
            model.cv,    
            model.X,
        ], axis=1)
    df = df.loc[df['fold'] >= 10]
    df[test_feature] = data.df[test_feature]
    df['current_model'] = model.yhat_proba(df.drop(columns=[model.y.name, model.cv.name, test_feature]))
    df['test_model'] = model2.yhat_proba(df.drop(columns=[model.y.name, model.cv.name]))

    bins = pd.qcut(df[test_feature], n_bins, duplicates='drop')
    df['bin'] = bins
    df['bin'] = df['bin'].astype(str)

    count_by_level = df.groupby('bin').count()[['hit_count']].reset_index().rename(columns={'hit_count': 'count'})

    ave_by_level = (
        df.groupby('bin')
        .mean()[['hit_count', 'current_model', 'test_model']]
        .reset_index()
        .set_index('bin')
        .join(
            count_by_level.set_index('bin'),
            on='bin'
        ).reset_index()
        .sort_values('bin', ascending=True)
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=ave_by_level['bin'],
            y=ave_by_level['count'],
            name='Count',
            marker=dict(
                color='gray',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            # do not show legend for this trace
            showlegend=False
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level['bin'],
            y=ave_by_level['hit_count'],
            mode='lines+markers',
            name='Actual',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            line=dict(
                color='blue',
                width=2
            ),

        ),
            secondary_y=False
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level['bin'],
            y=ave_by_level['current_model'],
            mode='lines+markers',
            name='Current Model',
            marker=dict(
                size=10,
                color='red',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),

            line=dict(
                color='red',
                width=1.5,
                dash='dashdot'
            ),
        ),
            secondary_y=False
    )

    fig.add_trace(
        go.Scattergl(
            x=ave_by_level['bin'],
            y=ave_by_level['test_model'],
            mode='lines+markers',
            name='Test Model',
            marker=dict(
                size=7,
                color='green',
                opacity=0.5,
                line=dict(
                    color='black',
                    width=0.5
                )
            ),
            line=dict(
                color='green',
                width=1.5,
                dash='dashdot'
            ),
        ),
            secondary_y=False
    )

    fig.update_layout(
        title='Actual vs. Model Predictions',
        xaxis_title=test_feature,
        yaxis_title='Hit Ratio',
        showlegend=True
    )

    fig.update_yaxes(title_text="Number of Quotes", secondary_y=True)

    return fig

import polars as pl
def marginal_effect_distro(df:pl.LazyFrame, feature_name: str | None = None, by: str | None = None):
    import plotly.graph_objects as go
    import plotly.express as px

    fig = go.Figure()
    
    if feature_name is not None:
        df = df.filter(pl.col('term') == feature_name)
        
    cols_to_drop_for_customdata = [
        'rowid',
        'contrast',
        'std_error',
        'statistic',
        'p_value',
        'conf_low',
        's_value',
        'conf_high',
        'predicted',
        'predicted_lo',
        'predicted_hi',   
    ]
    
    customdata = df.drop(cols_to_drop_for_customdata).collect().to_pandas()
    
    for col in customdata.columns:
        if col.startswith('is_') or col.startswith('has_'):
            customdata[col] = customdata[col].round(0).astype(int)
        
        
    hovertemplate='<b>dy/d[%{customdata[0]}] = %{customdata[1]:.1%}</b><br>'
    for i, dterm in enumerate(customdata.columns.tolist()[2:]):
        hovertemplate += f"<br><b>{dterm}:</b> "
        hovertemplate += '%{customdata[' 
        hovertemplate += f'{i+2}'
        if customdata.iloc[:, i].dtype == float:
            hovertemplate += ']:.2f}'
        else:
            hovertemplate += ']}'
        
    params = dict(
        x=df.select('term').collect()['term'],
        y=df.select('estimate').collect()['estimate'],
        line=dict(
            width=1
        ),
        marker=dict(
            opacity=0.3,
            line=dict(width=0),
#             color='firebrick'
        ),
        customdata=customdata,
        hovertemplate=hovertemplate
        
    )
    
    if by is None:
        fig.add_trace(
            go.Violin(**params) if feature_name is not None else go.Box(**params)
        )
    else:
        for i, level in enumerate(df.select(pl.col(by)).unique().collect()[by].to_list()):
            params = dict(
                x=df.filter(pl.col(by)==level).select(by).collect()[by],
                y=df.filter(pl.col(by)==level).select('estimate').collect()['estimate'],
                line=dict(
                    width=1
                ),
                marker=dict(
                    opacity=0.3,
                    line=dict(width=0),
                    color=px.colors.qualitative.G10[i]
                ),
                customdata=customdata.loc[customdata[by] == level],
                hovertemplate=hovertemplate,
                name=level,
                legendgroup=f"Level of {by}"
            )
            
            fig.add_trace(
                go.Violin(**params) if feature_name is not None else go.Box(**params)
            )
        
    
    if feature_name is not None:
        layout=dict(
            title=dict(text=f"Distribution of marginal effects of [{feature_name}]"),
            yaxis=dict(
                title=f"<b>Hit ratio impact</b><br>Unit increase to <i>{feature_name}</i>"
            )
        )
    else:
        layout=dict(
            title=dict(text=f"Distribution of marginal effects of all features"),
            yaxis=dict(
                title=f"<b>Hit ratio impact</b><br>Unit increase to <i>each feature</i>"
            )
        )
        
    layout['width'] = 900
    layout['height'] = 800
    fig.update_layout(**layout)
    return fig