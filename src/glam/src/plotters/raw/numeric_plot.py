def numeric_plot(model, test_feature, n_bins=5):
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