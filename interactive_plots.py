"""
Created on Tue Mar  3 13:53:58 2020

@author: xf18155
"""

import plotly.express as px
from plotly.offline import plot
import pandas as pd
import random
import re
import os


def plot_data(df, base_path, plot_name, mode):
    # df = px.data.gapminder()
    print(df.head())

    fig = px.scatter(df, x="new_x", y="new_y", hover_data=["popularity", "name"],
                     size="sizes", color="cluster",
                     hover_name="name", size_max=80, template="plotly_white")
    if mode == "I":
        title = 'Results from data mining on CORD-19 dataset: top mined interventions for Covid-19 '
    elif mode == "P":
        title = 'Results from data mining on CORD-19 dataset: top mined populations for Covid-19 '
    elif mode == "C":
        title = 'Results from data mining on CORD-19 dataset: top mined health conditions for Covid-19 '

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title=''
        )
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
    )

    plot(fig, filename=os.path.join(base_path, plot_name + ".html"))  # make, and save plot
    df = df.drop(["new_x", "new_y", "sizes"], axis=1)
    df.to_csv(os.path.join(base_path, plot_name + "_plot.csv"))


def make_centres(n_entries):
    coordinates = [(x - 0.5, y - 0.5) for x in range(1, n_entries + 1) for y in range(1, n_entries + 1)]
    random.shuffle(coordinates)
    #print(coordinates)
    return coordinates  # return coordinates


def make_plot(n_entries=7, mode="P"):
    # define paths to save outputs in current working dir

    plot_name = os.path.join(os.getcwd(), "predictions", 'plots', mode + "_plot_top_" + str(n_entries * n_entries))
    base_path = os.path.join(os.getcwd(), "predictions")
    df_map = pd.read_csv(os.path.join(base_path,
                                      mode + "_deduped.csv"))  # for the clustering, so that all deduped entities are lustered and plotted next to the "main"entity
    df_data = pd.read_csv(os.path.join(base_path, "predictionsLENA_" + mode + ".csv"))

    df_map = df_map[:n_entries * n_entries]  # get lines of interest
    print(df_map.shape)

    coordinates = make_centres(n_entries)

    new_x = []
    new_y = []
    name = []
    condition = []
    ids = []
    popularity = []
    sizes = []

    for index, row in df_map.iterrows():

        name.append(row[1])  #######data of central point
        condition.append(row[1])
        new_x.append(coordinates[index][0])
        new_y.append(coordinates[index][1])
        ids.append(" ".join(list(df_data.loc[df_data['Condition'] == row[1]]["Pubmed_Search"].values)))

        try:
            popularity.append(df_data.loc[df_data['Condition'] == row[1]]["Counts"].values[0])
            sizes.append(df_data.loc[df_data['Condition'] == row[1]]["Counts"].values[0])

        except:  # some of our grey/blue x entries are not explicitely existing as such, therefore they have no counts
            popularity.append(0)
            sizes.append(0)

        print(index)

        for c in re.split(";;", str(row[2])):  # data of sattelite points
            if c != "nan":  # c==nan means that now other name was found for this c. happens sometimes when mining on really small datasets

                if c[0] == " ":  # remove whitespace in front
                    c = c[1:]

                name.append(c)
                condition.append(row[1])

                if random.random() > 0.5:
                    new_x.append(coordinates[index][0] + random.uniform(0, 0.5))
                else:
                    new_x.append(coordinates[index][0] - random.uniform(0, 0.5))

                if random.random() > 0.5:
                    new_y.append(coordinates[index][1] + random.uniform(0, 0.5))
                else:
                    new_y.append(coordinates[index][1] - random.uniform(0, 0.5))

                ids.append(" ".join(list(df_data.loc[df_data['Condition'] == c]["Pubmed_Search"].values)))

                # print("---")
                # print(c)
                # print(df_data.loc[df_data['Condition'] == c]["Pubmed_Search"].values)
                # print(df_data.loc[df_data['Condition'] == c]["Counts"].values)
                nrs = df_data.loc[df_data['Condition'] == c]["Counts"].values[0]
                popularity.append(nrs)
                sizes.append(nrs)

    new_df = pd.DataFrame(list(zip(new_x, new_y, name, ids, popularity, condition, sizes)),
                          columns=["new_x", "new_y", "name", "ids", "popularity", "cluster", "sizes"])
    plot_data(new_df, base_path, plot_name, mode)


#make_plot(n_entries=7, mode="P")
#################################################################################################################################################
#
#KAGGLE version here, we need different libraries to plot in a notebook!
#
#
########################################
import plotly.express as px
# from plotly.offline import plot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import pandas as pd
import random
import re
import os


def plot_data_kaggle(df, base_path, plot_name, mode):
    # df = px.data.gapminder()
    # print(df.head())

    fig = px.scatter(df, x="new_x", y="new_y", hover_data=["popularity", "name"],
                     size="sizes", color="cluster",
                     hover_name="name", size_max=80, template="plotly_white")
    if mode == "I":
        title = 'Results from data mining on CORD-19 dataset: top mined interventions for Covid-19 '
    elif mode == "P":
        title = 'Results from data mining on CORD-19 dataset: top mined populations for Covid-19 '
    elif mode == "C":
        title = 'Results from data mining on CORD-19 dataset: top mined health conditions for Covid-19 '

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title=''
        )
        # paper_bgcolor='rgb(243, 243, 243)',
        # plot_bgcolor='rgb(243, 243, 243)',
    )

    # plot(fig, filename=os.path.join(base_path, plot_name + ".html"))  # make, and save plot

    iplot(fig, filename=os.path.join(base_path, plot_name + ".html"))
    df = df.drop(["new_x", "new_y", "sizes"], axis=1)
    df.to_csv(os.path.join(base_path, plot_name + "_plot.csv"))


def make_plot_kaggle(n_entries=7, mode="P"):
    # define paths to save outputs in current working dir

    plot_name = os.path.join(os.getcwd(), "predictions", 'plots', mode + "_plot_top_" + str(n_entries * n_entries))
    base_path = os.path.join(os.getcwd(), "predictions")
    df_map = pd.read_csv(os.path.join(base_path,
                                      mode + "_deduped.csv"))  # for the clustering, so that all deduped entities are lustered and plotted next to the "main"entity
    df_data = pd.read_csv(os.path.join(base_path, "predictionsLENA_" + mode + ".csv"))

    df_map = df_map[:n_entries * n_entries]  # get lines of interest
    print("Plotting top {} mining results and saving plots and data to {}.. ".format(n_entries * n_entries, plot_name))

    coordinates = make_centres(n_entries)

    new_x = []
    new_y = []
    name = []
    condition = []
    ids = []
    popularity = []
    sizes = []

    for index, row in df_map.iterrows():

        name.append(row[1])  #######data of central point
        condition.append(row[1])
        new_x.append(coordinates[index][0])
        new_y.append(coordinates[index][1])
        ids.append(" ".join(list(df_data.loc[df_data['Condition'] == row[1]]["Pubmed_Search"].values)))

        try:
            popularity.append(df_data.loc[df_data['Condition'] == row[1]]["Counts"].values[0])
            sizes.append(df_data.loc[df_data['Condition'] == row[1]]["Counts"].values[0])

        except:  # some of our grey/blue x entries are not explicitely existing as such, therefore they have no counts
            popularity.append(0)
            sizes.append(0)

        # print(index)

        for c in re.split(";;", str(row[2])):  # data of sattelite points
            if c != "nan":  # c==nan means that now other name was found for this c. happens sometimes when mining on really small datasets

                if c[0] == " ":  # remove whitespace in front
                    c = c[1:]

                name.append(c)
                condition.append(row[1])

                if random.random() > 0.5:
                    new_x.append(coordinates[index][0] + random.uniform(0, 0.5))
                else:
                    new_x.append(coordinates[index][0] - random.uniform(0, 0.5))

                if random.random() > 0.5:
                    new_y.append(coordinates[index][1] + random.uniform(0, 0.5))
                else:
                    new_y.append(coordinates[index][1] - random.uniform(0, 0.5))

                ids.append(" ".join(list(df_data.loc[df_data['Condition'] == c]["Pubmed_Search"].values)))

                # print("---")
                # print(c)
                # print(df_data.loc[df_data['Condition'] == c]["Pubmed_Search"].values)
                # print(df_data.loc[df_data['Condition'] == c]["Counts"].values)
                nrs = df_data.loc[df_data['Condition'] == c]["Counts"].values[0]
                popularity.append(nrs)
                sizes.append(nrs)

    new_df = pd.DataFrame(list(zip(new_x, new_y, name, ids, popularity, condition, sizes)),
                          columns=["new_x", "new_y", "name", "ids", "popularity", "cluster", "sizes"])
    plot_data_kaggle(new_df, base_path, plot_name, mode)
