import pandas as pd


def mean_median_plot(csv):
    df = pd.read_csv(csv, header=0, parse_dates=['date'],
                     infer_datetime_format=True,
                     usecols=['date', 'mean', 'median'])
    output_file('mean_median.html')
    show(TimeSeries(df.sort('date'), title='Air Temperature',
                    index='date',
                    xlabel='Date', ylabel='Air Temperature (°C)', legend=True))


def heatmap(csv):
    import seaborn as sns
    from matplotlib import pyplot as plt
    df = pd.read_csv(csv, header=0, parse_dates=['date'],
                     infer_datetime_format=True)
    piv = (df.groupby(['row', 'col'], as_index=False)
             .temp
             .mean()
             .pivot('row', 'col', 'temp')
             .rename(columns=dict(row='Latitude', col='Longitude',
                                  temp='Average Temperature')))
    ax = sns.heatmap(piv)
    for i, (xlabel, ylabel) in enumerate(zip(ax.xaxis.get_ticklabels(),
                                             ax.yaxis.get_ticklabels())):
        should_show = bool(i % 50)
        xlabel.set_visible(should_show)
        ylabel.set_visible(should_show)

    plt.show()
