import os
import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import plotly.tools as tls
import numpy as np

import csv
import glob
from sklearn.preprocessing import LabelEncoder
from PIL import Image

idrpath = './p27kid/'      # Directory generated by create_mutational_scanning.py

zscores = np.array([l for l in csv.reader(open(idrpath + "zscores.txt"), delimiter="\t")])
sequence = np.load(idrpath + "aa_labels.npy")
sequence_no_num = [x.split("\n")[0] + x.split("\n")[1] for x in sequence]

features = zscores[1:][np.argsort(-zscores[1:, 1].astype(np.float32)), 0][:5]
feature_dict = {}
traces = []
n = 0
min_avg = 0
for f in features:
    if "average" in f:
        vector = np.sum(np.load(idrpath + f + "_heat_map.npy"), axis=1)
        curr_sequence = [x + "\n Average " + f.split("_")[0] for x in sequence]
        traces.append(go.Bar(x=np.arange(len(vector)), y=vector, name="Average " + f.split("_")[0], hoverinfo="text",
                             hovertext=curr_sequence, opacity=0.5, marker={'color': "#61d2dc"}))
        feature_dict["Average " + f.split("_")[0]] = n
        n += 1
        if np.min(vector) < min_avg:
            min_avg = np.min(vector)
for f in features:
    if "max" in f:
        vector = np.sum(np.load(idrpath + f + "_heat_map.npy"), axis=1)
        vector[np.where(np.abs(vector) < 10)] = 0
        vector[np.where(np.abs(vector) >= 10)] = 0.2
        curr_sequence = [x + "\n Max " + f.split("_")[0] for x in sequence]
        traces.append(go.Bar(x=np.arange(len(vector)), y=vector, name="Max " + f.split("_")[0], hoverinfo="text",
                             hovertext=curr_sequence, opacity=0.5, marker={'color': "#df5758"}, base=min_avg - 0.5))
        feature_dict["Max " + f.split("_")[0]] = n
        n += 1
feature_dict["All Features"] = -1

amino_acids = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']
enc = LabelEncoder().fit(amino_acids)
amino_acids = enc.inverse_transform(np.arange(0, 20))
sorted_indices = [5, 15, 16, 19, 1, 13, 11, 8, 14, 6, 2, 3, 0, 17, 9, 7, 12, 18, 4, 10]
amino_acids = amino_acids[sorted_indices]

app = dash.Dash()
app.layout = html.Div(children=[
    html.H1(children='Mutational Scanning Summary for hnRNP A1 IDR (182 to 372)',
            style={'textAlign': 'center', 'font-family': "Helvetica"}),
    html.Div(
        [
            dcc.Dropdown(
                id="FeatureDropdown",
                options=[{
                    'label': i,
                    'value': i
                } for i in feature_dict.keys()],
                value='All Features', clearable=False),
        ],
        style={'width': '25%', 'display': 'inline-block'}),
    dcc.Graph(
        id='summary',
        figure={
            'data': traces,
            'layout': go.Layout(barmode='overlay',
                                yaxis=dict(title="Favorability of Residue", visible=True, autorange='reversed'),
                                xaxis=dict(title="Position", visible=True, linewidth=1), hovermode='closest', bargap=0.0)}),
    dcc.Loading(id='lettermap', children=[]),
    dcc.Loading(id='heatmap', children=[])
])

@app.callback(
    dash.dependencies.Output('summary', 'figure'),
    [dash.dependencies.Input('FeatureDropdown', 'value')])
def update_graph(value):
    if value == "All Features" or "":
        return {
            'data': traces,
            'layout': go.Layout(barmode='overlay',
                                yaxis=dict(title="Favorability of Residue", visible=True, autorange='reversed'),
                                xaxis=dict(title="Position", visible=True, linewidth=1), hovermode='closest', bargap=0.0)
        }
    else:
        if "Average" in value:
            f_index = feature_dict[value]
            return {
                'data': [traces[f_index]],
                'layout': go.Layout(barmode='overlay',
                                    yaxis=dict(title="Favorability of Residue", visible=True, autorange='reversed'),
                                    xaxis=dict(title="Position", visible=True, linewidth=1), hovermode='closest', showlegend=False, bargap=0.0)
            }
        else:
            f_index = feature_dict[value]
            return {
                'data': [traces[f_index]],
                'layout': go.Layout(barmode='overlay',
                                    yaxis=dict(title="Favorability of Residue", visible=False, autorange=False,
                                               range=[min_avg - 0.1, min_avg - 0.7], dtick=0.1),
                                    xaxis=dict(title="Position", visible=True, linewidth=1), hovermode='closest', showlegend=False, bargap=0.0)
            }

@app.callback(
    dash.dependencies.Output('heatmap', 'children'),
    [dash.dependencies.Input('FeatureDropdown', 'value')])
def update_heatmap(value):
    if value == "All Features" or "":
        pass
    elif "Average" in value:
        curr_heatmap = np.load(idrpath + value.split(" ")[1] + "_average_heat_map.npy").T
        fig = go.Figure(data=go.Heatmap(z=curr_heatmap,
                                        x=sequence_no_num,
                                        y=amino_acids,
                                        colorscale='RdBu', reversescale=True, zmid=0.0, colorbar=dict(thickness=10),
                                        hovertemplate='Wild-Type Amino Acid: %{x}<br>Mutated Amino Acid: %{y}<br>Change in Feature: %{z:.3f}<extra></extra>'))
        fig.update_layout(title="Mutational Scanning Heat Map for " + value,
                          yaxis_nticks=len(amino_acids), yaxis=dict(visible=True, autorange='reversed'))
        return html.Div([dcc.Graph(id='heatmap_graph', figure=fig)])
    else:
        curr_heatmap = np.load(idrpath + value.split(" ")[1] + "_max_heat_map.npy").T
        fig = go.Figure(data=go.Heatmap(z=curr_heatmap,
                                        x=sequence_no_num,
                                        y=amino_acids,
                                        colorscale='RdBu', reversescale=True, zmid=0.0, colorbar=dict(thickness=10),
                                        hovertemplate='Wild-Type Amino Acid: %{x}<br>Mutated Amino Acid: %{y}<br>Change in Feature: %{z}<extra></extra>'))
        fig.update_layout(title="Mutational Scanning Heat Map for " + value,
                          yaxis_nticks=len(amino_acids), yaxis=dict(visible=True, autorange='reversed'))
        return html.Div([dcc.Graph(id='heatmap_graph', figure=fig)])

@app.callback(
    dash.dependencies.Output('lettermap', 'children'),
    [dash.dependencies.Input('FeatureDropdown', 'value')])
def update_heatmap(value):
    if value == "All Features" or "":
        pass
    elif "Average" in value:
        image = Image.open(idrpath + value.split(" ")[1] + "_average_letter_map.png")
        npimage = np.array(image)
        w = npimage.shape[0]
        h = npimage.shape[1]
        scalefactor = 1.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, h * scalefactor],
                y=[0, w * scalefactor],
                mode="markers",
                marker_opacity=0))
        fig.update_xaxes(
            visible=False,
            range=[0, h * scalefactor])
        fig.update_yaxes(
            visible=False,
            range=[0, w * scalefactor],
            scaleanchor="x")
        fig.add_layout_image(
            dict(
                x=0,
                sizex=h * scalefactor,
                y=w * scalefactor,
                sizey=w * scalefactor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=image)
        )
        # Configure other layout
        fig.layout.update(
            margin={"b": 0},
            title="Mutational Scanning Letter Map for " + value,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return html.Div([dcc.Graph(id='lettermap_graph', figure=fig)])
    else:
        image = Image.open(idrpath + value.split(" ")[1] + "_max_letter_map.png")
        npimage = np.array(image)
        w = npimage.shape[0]
        h = npimage.shape[1]
        scalefactor = 1.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, h * scalefactor],
                y=[0, w * scalefactor],
                mode="markers",
                marker_opacity=0))
        fig.update_xaxes(
            visible=False,
            range=[0, h * scalefactor])
        fig.update_yaxes(
            visible=False,
            range=[0, w * scalefactor],
            scaleanchor="x")
        fig.add_layout_image(
            dict(
                x=0,
                sizex=h * scalefactor,
                y=w * scalefactor,
                sizey=w * scalefactor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=image)
        )
        # Configure other layout
        fig.layout.update(
            margin={"b": 0},
            title="Mutational Scanning Letter Map for " + value,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return html.Div([dcc.Graph(id='lettermap_graph', figure=fig)])

if __name__ == '__main__':
    app.run_server(debug=True)
