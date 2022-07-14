#!/usr/bin/env python3

"""
Relion analyse dashboard
Rafael Fernandez-Leiro & Nayim Gonzalez-Rodriguez 2022
"""

"""
Activate conda environment before running
Usage: run relion_analyse.py in your relion project directory
"""


### Libraries setup
import os
import pandas as pd
import numpy as np
import starfile
import pathlib
import dash
from dash import html
from dash import dcc
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import socket
import argparse
import glob
import seaborn as sns
from sklearn.cluster import KMeans
import dash_cytoscape as cyto
import regex as re

# Load extra layouts for pipeline display
cyto.load_extra_layouts()

# Parsing port number and host
parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", help = "choose port to run the webapp")
parser.add_argument("--host", "-host", help = "choose host to run the webapp")
parser.add_argument("--debug", "-d", help = "launch app in debug mode")
args, unknown = parser.parse_known_args()


# Set localhost and 8051 as host and port by default
if not args.port: port_number = 8051
else: port_number = args.port
if not args.host: hostname = socket.gethostname()
else: hostname = args.host 
if not args.debug: debug_mode = False
else: debug_mode = args.host


### FUNCTION DEFINITIONS ###

# Reading pipeline data
def pipeline_reader1(pipeline_starfile, nodetype):
    pipeline_df = starfile.read(pipeline_starfile)['pipeline_nodes']
    nodes = list(pipeline_df[pipeline_df['rlnPipeLineNodeTypeLabel'].str.contains(str(nodetype), na=False)]['rlnPipeLineNodeName'])
    return nodes

def pipeline_reader2(pipeline_starfile, nodetype):
    pipeline_df = starfile.read(pipeline_starfile)['pipeline_processes']
    nodes = list(pipeline_df[pipeline_df['rlnPipeLineProcessName'].str.contains(str(nodetype), na=False)]['rlnPipeLineProcessName'])
    return nodes

# Plot scatterplot with side violin plots 
def plot_scatter(dataframe, x_data, y_data, coloring):
    plot = px.scatter(
        data_frame=dataframe,
        x = x_data,
        y = y_data,
        marginal_x="violin",
        marginal_y="violin",
        color = coloring,
        render_mode = 'webgl',
        template = 'plotly_white',
        opacity = 0.5
        )
    return plot

# Plot scatterplot with side violin plots using WebGL for large datasets
def plot_scattergl(data_frame, x_data, y_data, coloring, colorscale, hist_color):
    plot = go.Figure()
    main_scatter = plot.add_scattergl(
        x = data_frame[x_data],
        y = data_frame[y_data],
        mode = 'markers',
        marker_colorscale=colorscale,
        marker = dict(color=coloring, opacity=0.5),
        hoverinfo='skip',
        fillcolor = 'white'
    )
    side_histogram_x = plot.add_trace(go.Violin(
        x = data_frame[x_data],
        name = x_data,
        yaxis = 'y2',
        marker = dict(opacity=0.5, color=hist_color),
    ))
    side_histogram_y = plot.add_trace(go.Violin(
        y = data_frame[y_data],
        name = y_data,
        xaxis = 'x2',
        marker = dict(opacity=0.5, color=hist_color),
    ))
    plot.layout = dict(xaxis=dict(domain=[0, 0.85], zeroline=True, title=x_data,gridcolor='#CBCBCB'),
                yaxis=dict(domain=[0, 0.85], zeroline=True, title=y_data, gridcolor='#CBCBCB'),
                showlegend=False,
                margin=dict(t=50),
                hovermode='closest',
                bargap=0,
                xaxis2=dict(domain=[0.85, 1], showgrid=True, zeroline=False),
                yaxis2=dict(domain=[0.85, 1], showgrid=True, zeroline=False),
                height=600,
                plot_bgcolor = 'rgba(0,0,0,0)',
    )
    
    def do_zoom(layout, xaxis_range, yaxis_range):
        inds = ((xaxis_range[0] <= data_frame[x_data]) & (data_frame[x_data] <= xaxis_range[1]) &
                (yaxis_range[0] <= data_frame[y_data]) & (data_frame[y_data] <= yaxis_range[1]))
        with plot.batch_update():
            side_histogram_x.x = data_frame[x_data][inds]
            side_histogram_y.y = data_frame[y_data][inds]

    plot.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')

    return plot

# Plotting line plots
def plot_line(data_frame, x_data, y_data):
    plot = px.line(
        x = x_data,
        y = data_frame[y_data],
        render_mode = 'webgl',
        template = 'plotly_white',
    )
    return plot

# Area plots for class distribution over iterations
def plot_area(data_frame):
    plot = px.area(data_frame, template = 'plotly_white')
    return plot

# Heatmap plots for angular distribution
def plot_angdist(data_frame, x_data, y_data, bins, coloring):
    plot = px.density_heatmap(
        data_frame = data_frame,
        x = x_data,
        y = y_data,
        facet_col = coloring,
        nbinsx = bins,
        nbinsy = bins
    )
    return plot


### STYLE ###

# Header
header_style = {'width':'100%', 'vertical-align':'center' , 'display':'inline-flex', 'justify-content': 'space-between'}
title1_style = {"margin-left": "15px", "margin-top": "15px", "margin-bottom": "0em", "color": "Black",
                "font-family" : "Helvetica", "font-size":"2.5em"}
title2_style = {"margin-left": "15px", "margin-top": "0em", "color": "Black", "font-family" : "Helvetica"}
header_button_style = {'margin-top':'40px','margin-right':'40px'}

# Tabs
tabs_style = {'height': '3em', 'width': '100%', 'display': 'inline', 'vertical-align': 'bottom', 'borderBottom':'3px #000000'}
tab_style = {'padding':'0.5em', "font-family" : "Helvetica", 'background-color':'white'}
tab_selected_style = {'padding':'0.5em', 'borderTop': '3px solid #000000', "font-family" : "Helvetica", 'font-weight':'bold'}
tab_left_div_style = {'width':'20%', 'vertical-align':'top' , 'display':'inline-block'}
tab_right_div_style = {'width':'80%', 'vertical-align':'top' , 'display':'inline-block'}
tab_bottom_right_style = {'width':'100%', 'vertical-align':'top' , 'display':'inline-block'}
H5_title_style = {'font-family':'Helvetica', 'font-weight':'regular'}
pre_style = {'font-size':'0.65em', 'white-space':'pre-line' ,'font-family':'Helvetica', 'font-weight':'regular'}

# Dropdowns
dd_style = {"font-size":"0.9em",'width':"100%", "margin-left": "0%", "color": "black",
            "font-family" : "Helvetica", 'vertical-align': 'top', "margin-bottom":"2px"}
box_style = {"font-size":"0.9em",'padding':'0.3em 1.2em','width':"87%", "margin-left": "0%", "color": "black",
            "font-family" : "Helvetica", 'vertical-align': 'center', "margin-bottom":"2px"}

# Buttons
bt_style = {"align-items": "center", "background-color": "#F2F3F4", "border": "2px solid #000",
            "box-sizing": "border-box", "color": "#000", "cursor": "pointer", "display": "inline-flex",
            "font-family": "Helvetica", "margin-bottom":"3px", 'padding':'0.3em 1.2em', 'margin':'0 0.3em 0.3em 0',
            "font-size": "0.9em", 'font-weight':'500', 'padding':'0.3em 1.2em', 'border-radius':'2em', 'text-align':'center',
            'transition':'all 0.2s'}

# Pipeline nodes
nodes_style = [
        {'selector':'node', 'style':{'content':'data(label)','font-family':'monospace','text-wrap':'wrap',
                                    'text-max-width':'80px','text-overflow-wrap':'anywhere',
                                    'text-halign': 'center','text-valign': 'center','color': 'white',
                                    'text-outline-width': 2,'width':'90px', 'height':'90px'}},
        {'selector':'.Import', 'style':{'background-color':'white', 'shape':'round-rectangle',
                                        'border-style':'dashed', 'border-color':'black',
                                        'border-width':'4px', 'width':'140px'}},
        {'selector':'.AutoPick', 'style':{'background-color':'cornflowerblue'}},
        {'selector':'.Class3D', 'style':{'background-color':'#bb342f'}},
        {'selector':'.Class2D', 'style':{'background-color':'#dda448'}},
        {'selector':'.Select', 'style':{'background-color':'darkseagreen', 'width':'120px', 'height':'40px'}},
        {'selector':'.Extract', 'style':{'background-color':'white', 'shape':'square',
                                         'border-color':'black', 'border-width':'4px'}},
        {'selector':'.Refine3D', 'style':{'background-color':'#8d6a9f'}},
        {'selector':'.PostProcess', 'style':{'background-color':'white', 'line-color':'white', 'shape':'star',
                                             'border-color':'black', 'border-width':'4px'}},
]


### Initialising dash APP ###
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "RELION Analyse Dashboard"
server = app.server

### APP Layout ###
def serve_layout():
    return html.Div([
        # Title
        html.Div(style=header_style,children=[
            html.Div(children=[
                html.H1("RELION", style=title1_style),
                html.H2("Analyse Dashboard", style=title2_style),
                ]),
            html.Div(style=header_button_style, children=[
                html.A(html.Button('Reload pipeline', style=bt_style), id='RefreshPage', href='/'),
                ]),
            ]),

        # Tabs

        html.Div([
            dcc.Tabs([
                # Tab Pipeline

                dcc.Tab(label=' Relion Pipeline', children=[
                    html.Div(style={'width':'10%'}, children=[
                        # Reload Graph button
                    html.A(html.Button('Reload graph', style=bt_style), id='reloadgraphbutton', href='/'),
                        ]),
        
                    html.Div(children=[   
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            cyto.Cytoscape(id='pipeline_graph', layout={'name': 'dagre'},
                                           style={'width': '100%', 'height': '800px'},
                                           boxSelectionEnabled=True,
                                           elements=[],
                                           stylesheet=nodes_style),
                        ]),
                    ]),
                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse Micrographs

                dcc.Tab(label=' Analyse Micrographs', children=[
                    html.Div(style=tab_left_div_style, children=[
                        # Dropdowns for starfile selection
                        html.H5('Micrograph starfile to analyse', style=H5_title_style),
                        dcc.Dropdown(id='mic_star', placeholder='choose starfile...', options=pipeline_reader1('default_pipeline.star', 'MicrographsData'), style=dd_style),
                        # Dropdowns for variable selection
                        html.H5('Choose axes (x, y and colouring)', style=H5_title_style),
                        html.Div(children=[
                            dcc.Dropdown(id='mic_dropdown_x', value='rlnAccumMotionEarly', options=[], style=dd_style),
                            dcc.Dropdown(id='mic_dropdown_y', value='rlnAccumMotionLate', options=[], style=dd_style),
                            dcc.Dropdown(id='mic_dropdown_class', value='rlnOpticsGroup', options=[], style=dd_style),
                            dcc.RadioItems(id='mic-color-mode', value='None', options=['None','Discrete', 'Continuous']),
                        ]),

                        # Cluster number input and clustering button
                        html.H5("Cluster Micrographs (number of clusters):", style=H5_title_style),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='number_of_clusters_mic', value='2',style=box_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Cluster', id='clustering_button_mic', style=bt_style),
                            ]),
                        ]),    

                        # Selected micrographs print and export button
                        html.H5("Export selection (basename):", style={'font-family':'Helvetica', 'font-weight':'regular'}),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='basename_export_mic', value='exported', style=box_style),
                            ]),
                            html.Div(style={'width':'20%'}, children=[
                                html.Button('Export', id='export_micrographs_button', style=bt_style),
                            ]),
                            html.Div(style={'width':'20%'}, children=[
                                html.Button('Display', id='display_sel_mic', style=bt_style),
                            ]),
                        ]),
                        # Selected micrographs list                    
                        html.Div(style={'width':'95%','margin-right':'1%','margin-left':'1%'},children=[                    
                            html.Pre(id='selected_micrographs' , style=pre_style),
                        ]),
                    ]),
        
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            dcc.Graph(id='mic_scatter2D_graph', figure={}, style={'display': 'inline-block', 'width': '80%', 'height': '70vh'}),
                        ]),
                    ]),
                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse Particles

                dcc.Tab(label=' Analyse Particles', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns for starfile selection
                        html.H5('Particle starfile to analyse', style=H5_title_style),
                        dcc.Dropdown(id='ptcl_star', placeholder='choose starfile...', options=pipeline_reader1('default_pipeline.star', 'ParticlesData'), style=dd_style),
            
                        # Dropdowns for variable selection
                        html.H5('Choose axes (x, y and colouring)', style={'font-family':'Helvetica', 'font-weight':'regular'}),
                        html.Div(children=[
                            dcc.Dropdown(id='ptcl_dropdown_x', value='rlnCtfMaxResolution', options=[], style=dd_style),
                            dcc.Dropdown(id='ptcl_dropdown_y', value='rlnAutopickFigureOfMerit', options=[], style=dd_style),
                            dcc.Dropdown(id='ptcl_dropdown_class', value='rlnOpticsGroup', options=[], style=dd_style),
                        ]),

                        # Cluster number input and clustering button
                        html.H5("Cluster Particles (number of clusters):", style=H5_title_style),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='number_of_clusters_ptcl', value='2',style=box_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Cluster', id='clustering_button_ptcl', style=bt_style),
                            ]),
                        ]),    

                        # Selected particles print and export button
                        html.H5("Export selection (basename):", style=H5_title_style),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='basename_export_ptcl', value='exported', style=box_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Export', id='export_particles_button', style=bt_style),
                            ]),
                        ]),
                        # Selected particles list                    
                        html.Div(style={'width':'95%','margin-right':'1%','margin-left':'1%'},children=[                    
                            html.Pre(id='selected_particles' , style=pre_style),
                        ]),    
                    ]),
        
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            dcc.Graph(id='ptcl_scatter2D_graph', figure={}, style={'display': 'inline-block', 'width': '80%', 'margin-lef':'15px', 'height': '70vh'}),
                        ]),
                    ]),

                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 2D Classification

                dcc.Tab(label=' Analyse 2D Classification', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Classification job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='C2Djob2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Class2D'), style=dd_style),
                        dcc.Input(id='C2Dfollow_msg', type='text', debounce=True, style=box_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Dropdown(id='C2Dfollow_dropdown_y', value='rlnChangesOptimalClasses', options=[], style=dd_style),
                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display classes (RELION)', id='C2Ddisplay_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'50%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='C2Dfollow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'50%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow classes
                            dcc.Graph(id='C2Dclassnumber_graph', figure={}),
                        ]),
                    ])], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 3D Classification

                dcc.Tab(label=' Analyse 3D Classification', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Classification job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='job2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Class3D'), style=dd_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Input(id='follow_msg', type='text', debounce=True, style=box_style),
                        dcc.Dropdown(id='follow_dropdown_y', value='rlnChangesOptimalClasses', options=[], style=dd_style),
                        dcc.Dropdown(id='follow_model_dropdown_y', value='rlnSpectralOrientabilityContribution', options=[], style=dd_style),

                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display classes (RELION)', id='display_last_ite', style=bt_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display maps (Chimera)', id='display_chimera_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='follow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow classes
                            dcc.Graph(id='classnumber_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                        # Plot FSC
                        dcc.Graph(id='follow_model', figure={})
                        ]),

                    ]),
                    html.Div(style=tab_bottom_right_style,children=[
                        html.Div(style={'width':'100%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='angdist_per_class', figure={}),
                        ]),
                    ])], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 3D Refinement

                dcc.Tab(label=' Analyse 3D Refinement', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Refine3D job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='ref_job2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Refine3D'), style=dd_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Input(id='ref_follow_msg', type='text', debounce=True, style=box_style),
                        dcc.Dropdown(id='ref_follow_dropdown_y', value='rlnCurrentResolution', options=[], style=dd_style),
                        dcc.Dropdown(id='ref_follow_model_dropdown_y', value='rlnGoldStandardFsc', options=[], style=dd_style),
                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display slices (RELION)', id='ref_display_last_ite', style=bt_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display maps (Chimera)', id='ref_display_chimera_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                        dcc.Graph(id='ref_follow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot FSC
                        dcc.Graph(id='ref_follow_model', figure={})
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                        dcc.Graph(id='ref_follow_angdist', figure={}),
                        ]),

                    ])], style=tab_style, selected_style=tab_selected_style),
                
                ], style=tabs_style),
        ])
    ])

app.layout = serve_layout

### Callbacks

## Callback reload pipeline
@app.callback(
    [Output(component_id='RefreshPage', component_property='title')],
    [Input(component_id='RefreshPage', component_property='n_clicks')],
    prevent_initial_call=True)

def reload_pipeline(pipeline_reload_button_press):

    pipeline_reload_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'pipeline_reload_button' in pipeline_reload_button_press_changed:
        print('reloading pipeline')
        bttitle = ''
        
    return ([bttitle])

## Callback Pipeline Graph
@app.callback(
    [Output(component_id='pipeline_graph', component_property='elements')],
    [Input(component_id='reloadgraphbutton', component_property='n_clicks')],
    prevent_initial_call=False)

def plot_pipeline(reloadgraphbutton):

    pipe_nodes = starfile.read('default_pipeline.star')['pipeline_processes']['rlnPipeLineProcessName']
    pipe_input_edges = starfile.read('default_pipeline.star')['pipeline_input_edges']

    reloadgraphbutton_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'reloadgraphbutton' in reloadgraphbutton_press_changed:
        print('reloading graph')
    
    cytoelements = []
    for i in pipe_nodes:
        cytoelements.append({'data': {'id':i[-7:-1], 'label':i}, 'classes':i[:-8]})

    for i, row in pipe_input_edges.iterrows():
        try: cytoelements.append({'data': {'source':re.search(r'job\d\d\d', row['rlnPipeLineEdgeFromNode'])[0], 'target':row['rlnPipeLineEdgeProcess'][-7:-1]}})
        except:
            print('there\'s a problem with node: '+row['rlnPipeLineEdgeFromNode'])
            continue

    return ([cytoelements])

## Callback Micrographs
@app.callback(
    [Output(component_id='mic_dropdown_x', component_property='options'),
     Output(component_id='mic_dropdown_y', component_property='options'),
     Output(component_id='mic_dropdown_class', component_property='options'),
     Output(component_id='mic_scatter2D_graph', component_property='figure'),
     Output(component_id='selected_micrographs', component_property='children')],
    [Input(component_id='mic_star', component_property='value'),
     Input(component_id='mic-color-mode', component_property='value'),
     Input(component_id='mic_dropdown_x', component_property='value'),
     Input(component_id='mic_dropdown_y', component_property='value'),
     Input(component_id='mic_dropdown_class', component_property='value'),
     Input(component_id='mic_scatter2D_graph', component_property='selectedData'),
     Input(component_id='basename_export_mic', component_property='value'),
     Input(component_id='export_micrographs_button', component_property='n_clicks'),
     Input(component_id='display_sel_mic', component_property='n_clicks'),
     Input(component_id='number_of_clusters_mic', component_property='value'),
     Input(component_id='clustering_button_mic', component_property='n_clicks'),
     ]
)

def load_df_and_graphs(mic_star, mic_color_mode, mic_dd_x, mic_dd_y, mic_dd_class,
                       selectedMicData, basename_export_mic,
                       exportMic_button_press,display_sel_mic_button_press,
                       n_clusters_mic,clusteringMic_button_press):

### Micrographs

    # Importing CTF starfile data
    try:
        ctf_df_optics = starfile.read(mic_star)['optics']
        ctf_df = starfile.read(mic_star)['micrographs']
    except:
        print('No starfile selected')
        raise PreventUpdate

    # Importing MotionCorr data from all CTF-corrected micrographs, even if they
    # come from different motioncor jobs.
    motion_df = ''
    for i in ctf_df['rlnMicrographName'].str[:18].unique():
        motion_star_path = str(i)+'/corrected_micrographs.star'
        if type(motion_df) == type('') :
            motion_df = starfile.read(motion_star_path)['micrographs']
        else:
            motion_df = pd.concat([motion_df , starfile.read(motion_star_path)['micrographs']], ignore_index = True)

    # Merging CTF and MotionCorr data in a single dataframe for easy plotting
    job_df = pd.merge(ctf_df, motion_df) # what if they don't match?

    mic_dropdown_x = list(job_df)
    mic_dropdown_y = list(job_df)
    mic_dropdown_class = list(job_df)
   
    # Duplicating df info (really needed?)
    mic_dff = job_df.copy()

    # Color definitions
    color1 = 'cornflowerblue'

    # Clustering
    if clusteringMic_button_press:
        clustering_data_mic = pd.concat([mic_dff[mic_dd_x], mic_dff[mic_dd_y]], axis=1, keys=[mic_dd_x, mic_dd_y])
        clustering_values_mic = clustering_data_mic.values
        kmeans_mic = KMeans(n_clusters=int(n_clusters_mic), random_state=100)
        kmeans_mic.fit(clustering_values_mic)
        y_kmeans_mic = kmeans_mic.predict(clustering_values_mic)
        mic_dff['clusters'] = y_kmeans_mic
        mic_dropdown_class = list(mic_dff)

    # Coloring markers as discrete or continuous variables
    if mic_color_mode == 'None':
        mic_dd_class = None
    elif mic_color_mode == 'Discrete':
        mic_dd_class = mic_dff[mic_dd_class].astype(str)
    elif mic_color_mode == 'Continuous':
        mic_dd_class = mic_dff[mic_dd_class].astype(float)

    # Scatter plot
    mic_scatter2D = plot_scatter(mic_dff, mic_dff[mic_dd_x], mic_dff[mic_dd_y], mic_dd_class)
    mic_scatter2D.update_layout(),

    # Parsing info from manual on-plot selection
    selected_micrographs_indices = []
    NOTselected_micrographs_indices = []
    if isinstance(selectedMicData, dict):
        for i in selectedMicData['points']:
            selected_micrographs_indices.append(int(i['pointIndex']))
        NOTselected_micrographs = mic_dff.loc[mic_dff.index.difference(selected_micrographs_indices)]
        NOTselected_micrographs_indices = list(NOTselected_micrographs.index.values)

    # Output definitions
    selectionMic_output = 'You\'ve selected '+str(len(selected_micrographs_indices))+' micrographs with indices: '+str(selected_micrographs_indices)
    outfile_mic_YES = str(basename_export_mic + '_selected_micrographs.star')
    outfile_mic_NO = str(basename_export_mic + '_not_selected_micrographs.star')
    exportMic_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'export_micrographs_button' in exportMic_button_press_changed:
        dict_mic_output_YES = {'optics' : ctf_df_optics , 'micrographs' : ctf_df.iloc[selected_micrographs_indices]}
        dict_mic_output_NO = {'optics' : ctf_df_optics , 'micrographs' : ctf_df.iloc[NOTselected_micrographs_indices]}
        starfile.write(dict_mic_output_YES, outfile_mic_YES, overwrite=True)
        starfile.write(dict_mic_output_NO, outfile_mic_NO, overwrite=True)
        print('Exported selected micrographs as '+ outfile_mic_YES + ' and not selected micrographs as ' + outfile_mic_NO)

    # Display mics
    display_sel_mic_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'display_sel_mic' in display_sel_mic_button_press_changed:
        os.system(str('`which relion_display` --i '+outfile_mic_YES+' --gui '))

    ### RETURN

    return ([mic_dropdown_x, mic_dropdown_y, mic_dropdown_class, mic_scatter2D, selectionMic_output])

## Callback Particles
@app.callback(
    [Output(component_id='ptcl_dropdown_x', component_property='options'),
     Output(component_id='ptcl_dropdown_y', component_property='options'),
     Output(component_id='ptcl_dropdown_class', component_property='options'),
     Output(component_id='ptcl_scatter2D_graph', component_property='figure'),
     Output(component_id='selected_particles', component_property='children')],
    [Input(component_id='ptcl_star', component_property='value'),
     Input(component_id='ptcl_dropdown_x', component_property='value'),
     Input(component_id='ptcl_dropdown_y', component_property='value'),
     Input(component_id='ptcl_dropdown_class', component_property='value'),
     Input(component_id='ptcl_scatter2D_graph', component_property='selectedData'),
     Input(component_id='basename_export_ptcl', component_property='value'),
     Input(component_id='export_particles_button', component_property='n_clicks'),
     Input(component_id='number_of_clusters_ptcl', component_property='value'),
     Input(component_id='clustering_button_ptcl', component_property='n_clicks')]
)

def load_df_and_graphs(ptcl_star, ptcl_dd_x, ptcl_dd_y, ptcl_dd_class,selectedPtclData,basename_export_ptcl,exportPtcl_button_press,n_clusters_ptcl,clusteringPtcl_button_press):

### Particles

    # Importing particles starfile data

    try:
        ptcl_df = starfile.read(ptcl_star)['particles']
        ptcl_df_optics = starfile.read(ptcl_star)['optics']
    except:
        print('No starfile selected')
        raise PreventUpdate

    ptcl_dropdown_x = list(ptcl_df)
    ptcl_dropdown_y = list(ptcl_df)
    ptcl_dropdown_class = list(ptcl_df)

    # Duplicating df info (really needed?)
    ptcl_dff = ptcl_df.copy()

    # Color definitions
    color1 = 'cornflowerblue'

    # Clustering

    if clusteringPtcl_button_press:
        clustering_data_ptcl = pd.concat([ptcl_dff[ptcl_dd_x], ptcl_dff[ptcl_dd_y]], axis=1, keys=[ptcl_dd_x, ptcl_dd_y])
        clustering_values_ptcl = clustering_data_ptcl.values
        kmeans_ptcl = KMeans(n_clusters=int(n_clusters_ptcl), random_state=100)
        kmeans_ptcl.fit(clustering_values_ptcl)
        y_kmeans_ptcl = kmeans_ptcl.predict(clustering_values_ptcl)
        ptcl_dff['clusters'] = y_kmeans_ptcl
        ptcl_dropdown_class = list(ptcl_dff)
        #ptcl_dd_class = 'clusters'

    if ptcl_dd_class == None: ptcl_dd_class = 'cornflowerblue'
    else: ptcl_dd_class = ptcl_dff[ptcl_dd_class].astype(float)
    
    # Particles plot using WebGL given probably large dataframes
    ptcl_scatter2D = plot_scattergl(ptcl_dff, ptcl_dd_x, ptcl_dd_y, ptcl_dd_class, 'Viridis', 'cornflowerblue')
    
    # Parsing info from manual on-plot selection
    selected_particle_indices = []
    NOTselected_particle_indices = []
    if isinstance(selectedPtclData, dict):
        for i in selectedPtclData['points']:
            selected_particle_indices.append(int(i['pointIndex']))
        NOTselected_particles = ptcl_dff.loc[ptcl_dff.index.difference(selected_particle_indices)]
        NOTselected_particle_indices = list(NOTselected_particles.index.values)

    # Output definitions

    selectionPtcl_output = 'You\'ve selected '+str(len(selected_particle_indices))+' particles with indices: '+str(selected_particle_indices)

    outfile_ptcl_YES = str(basename_export_ptcl + '_selected_particles.star')
    outfile_ptcl_NO = str(basename_export_ptcl + '_not_selected_particles.star')

    exportPtcl_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'export_particles_button' in exportPtcl_button_press_changed:
        dict_ptcl_output_YES = {'optics' : ptcl_df_optics , 'particles' : ptcl_df.iloc[selected_particle_indices]}
        dict_ptcl_output_NO = {'optics' : ptcl_df_optics , 'particles' : ptcl_df.iloc[NOTselected_particle_indices]}
        starfile.write(dict_ptcl_output_YES, outfile_ptcl_YES, overwrite=True)
        starfile.write(dict_ptcl_output_NO, outfile_ptcl_NO, overwrite=True)
        print('Exported selected particles as '+ outfile_ptcl_YES + ' and not selected particles as ' + outfile_ptcl_NO)


    ### RETURN

    return ([ptcl_dropdown_x, ptcl_dropdown_y, ptcl_dropdown_class, ptcl_scatter2D, selectionPtcl_output])


## Callback 2D Classification
@app.callback(
    [Output(component_id='C2Dfollow_msg', component_property='value'),
     Output(component_id='C2Dfollow_graph', component_property='figure'),
     Output(component_id='C2Dclassnumber_graph', component_property='figure'),
     Output(component_id='C2Dfollow_dropdown_y', component_property='options')],
    [Input(component_id='C2Djob2follow', component_property='value'),
     Input(component_id='C2Dfollow_dropdown_y', component_property='value'),
     Input(component_id='C2Ddisplay_last_ite', component_property='n_clicks'),
     ]
)

def load_df_and_graphs(C2Djob2follow, C2Dfollow_dd_y,C2Ddisplay_last_ite_button_press):

### Follow 2D

    job = C2Djob2follow
    follow_dd_y = C2Dfollow_dd_y

    if 'Class2D' in str(job):

        C2Dfollow_msg = 'Following Class2D job: '+str(job)
        all_opt = glob.glob(os.path.join(job+'run_it*_optimiser.star'))
        all_opt.sort()
        stars_opt = []
        for filename in all_opt:
            optimiser_df = starfile.read(filename)
            stars_opt.append(optimiser_df)
        follow_opt_df = pd.concat(stars_opt, axis=0, ignore_index=True)
        C2Dfollow_dd_y_list = list(follow_opt_df)
        all_model = glob.glob(os.path.join(job+'run_it*_model.star'))
        all_model.sort()
        stars_classnumber = []
        for filename in all_model:
            classnumber_df = starfile.read(filename)['model_classes']
            classnumber_df = classnumber_df['rlnClassDistribution']
            stars_classnumber.append(classnumber_df)
        follow_classnumber_df = pd.concat(stars_classnumber, axis=1, ignore_index=True)
        follow_classnumber_df.index=follow_classnumber_df.index +1
        follow_classnumber_df = follow_classnumber_df.T
        number_of_iterations = len(follow_opt_df[follow_dd_y])

    else:
        C2Dfollow_msg = 'Nothing to do here...'
        C2Dfollow_dd_y_list = ['']
        follow_opt_df = pd.DataFrame()
        follow_opt_df['bla'] = []
        follow_classnumber_df = pd.DataFrame()
        follow_dd_y = 'bla'
        number_of_iterations = 1

    # Plot parameters over iterations

    C2Dfollow_graph = plot_line(follow_opt_df, list(range(number_of_iterations)), follow_dd_y)
    C2Dfollow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="Iteration",yaxis_title=C2Dfollow_dd_y)

    # Plot class distribution over iterations
    C2Dclassnumber_graph = plot_area(follow_classnumber_df)
    C2Dclassnumber_graph.update_layout(title_text='Class distribution', title_x=0.5,xaxis_title="Iteration",yaxis_title="Class proportion")

    print(C2Dfollow_msg)

    # Display last ite
    C2Ddisplay_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in C2Ddisplay_last_ite_button_press_changed:
        print('displaying ' + all_opt[-1])
        os.system(str('`which relion_display` --i '+ all_opt[-1] +' --gui '))

    ### RETURN

    return ([C2Dfollow_msg, C2Dfollow_graph, C2Dclassnumber_graph, C2Dfollow_dd_y_list])


## Callback 3D Classification
@app.callback(
    [Output(component_id='follow_msg', component_property='value'),
     Output(component_id='follow_graph', component_property='figure'),
     Output(component_id='follow_dropdown_y', component_property='options'),
     Output(component_id='classnumber_graph', component_property='figure'),
     Output(component_id='follow_model', component_property='figure'),
     Output(component_id='follow_model_dropdown_y', component_property='options'),
     Output(component_id='angdist_per_class', component_property='figure')],
    [Input(component_id='job2follow', component_property='value'),
     Input(component_id='follow_dropdown_y', component_property='value'),
     Input(component_id='follow_model_dropdown_y', component_property='value'),
     Input(component_id='display_last_ite', component_property='n_clicks'),
     Input(component_id='display_chimera_last_ite', component_property='n_clicks')
     ]
)

def load_df_and_graphs(job2follow, follow_dd_y, follow_model_dd_y, display_last_ite_button_press, display_last_ite_chimera_button_press):

### Follow 3D

    job = job2follow

    if 'Class3D' in str(job):

        follow_msg = 'Following Class3D job: '+str(job)
        all_opt = glob.glob(os.path.join(job+'run_it*_optimiser.star'))
        all_opt.sort()
        stars_opt = []
        for filename in all_opt:
            optimiser_df = starfile.read(filename)
            stars_opt.append(optimiser_df)
        follow_opt_df = pd.concat(stars_opt, axis=0, ignore_index=True)
        follow_opt_df_last_ite_model = follow_opt_df['rlnModelStarFile'].iloc[-1]
        follow_opt_df_last_ite_maps = str(follow_opt_df_last_ite_model[:-10]+'class*mrc')
        follow_dd_y_list = list(follow_opt_df)

        all_model = glob.glob(os.path.join(job+'run_it*_model.star'))
        all_model.sort()
        stars_classnumber = []
        for filename in all_model:
            classnumber_df = starfile.read(filename)['model_classes']
            classnumber_df = classnumber_df['rlnClassDistribution']
            stars_classnumber.append(classnumber_df)
        follow_classnumber_df = pd.concat(stars_classnumber, axis=1, ignore_index=True)
        follow_classnumber_df.index=follow_classnumber_df.index +1
        follow_classnumber_df = follow_classnumber_df.T

        last_ite_model = str(all_model[-1])
        last_ite_model_df = starfile.read(last_ite_model)['model_classes']
        last_ite_fsc_df = starfile.read(last_ite_model)['model_class_1']
        last_ite_fsc_df_list = list(last_ite_fsc_df)
        follow_opt_df_last_ite_model = last_ite_model_df['rlnReferenceImage']
        resolutions = list(last_ite_fsc_df['rlnResolution'])
        number_of_iterations = len(follow_opt_df[follow_dd_y])

        number_of_its_class3D = len(follow_classnumber_df)-1
        angdist_per_class = starfile.read(job+f'run_it{number_of_its_class3D:03d}_data.star')['particles'][['rlnClassNumber', 'rlnAngleRot', 'rlnAngleTilt']]
        sampling_angle_class = float(starfile.read(job+f'run_it{number_of_its_class3D:03d}_sampling.star')['sampling_general']['rlnPsiStep'])


    else:

        follow_msg = 'Nothing to do here...'
        follow_dd_y_list = ['']
        follow_opt_df = pd.DataFrame()
        follow_opt_df['bla'] = []
        follow_classnumber_df = pd.DataFrame()
        follow_dd_y = 'bla'
        angdist_per_class = pd.DataFrame()
        angdist_per_class['rlnAngleRot'] = ['']
        angdist_per_class['rlnAngleTilt'] = ['']
        angdist_per_class['rlnClassNumber'] = [''] 
        sampling_angle_class = 1
        follow_dd_y = 'bla'
        follow_model_dd_y = 'bla'
        last_ite_fsc_df = pd.DataFrame()
        last_ite_fsc_df['rlnResolution'] = []
        last_ite_fsc_df['bla'] = []
        last_ite_fsc_df_list = ['']
        resolutions = ['']
        number_of_iterations = 1


    follow_graph = plot_line(follow_opt_df, list(range(number_of_iterations)), follow_dd_y)
    follow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="Iteration",yaxis_title=follow_dd_y)

    follow_model = plot_line(last_ite_fsc_df, resolutions, follow_model_dd_y)
    follow_model.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="1/Resolution",yaxis_title=follow_model_dd_y)
    if follow_model_dd_y == 'rlnGoldStandardFsc': follow_model.add_hline(y=0.143, line_dash='dash', line_color="grey")
    elif follow_model_dd_y == 'rlnSsnrMap': follow_model.add_hline(y=1, line_dash='dash', line_color="grey")
    follow_model.update_layout(title_text=f'Iteration {number_of_iterations:03d}', title_x=0.5)

    
    classnumber_graph = plot_area(follow_classnumber_df)
    classnumber_graph.update_layout(title_text='Class distribution', title_x=0.5,xaxis_title="Iteration",yaxis_title="Class proportion")
    
    angdist_per_class_plot = plot_angdist(angdist_per_class, 'rlnAngleRot', 'rlnAngleTilt', int(360//(sampling_angle_class)), 'rlnClassNumber')

    angdist_per_class_plot.update_layout(
        xaxis = dict(tickmode= 'linear', dtick = 90),
        yaxis = dict(tickmode= 'linear', dtick = 90),
        title_text=f'Angular Distribution',
        title_x=0.5
    )

    print(follow_msg)

    # Display last ite
    display_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in display_last_ite_button_press_changed:
        print('displaying ' + all_opt[-1])
        os.system(str('`which relion_display` --i '+ all_opt[-1] + ' --gui '))

    display_last_ite_chimera_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_chimera_last_ite' in display_last_ite_chimera_button_press_changed:
        print('displaying ' + ' '.join(follow_opt_df_last_ite_model) + ' in chimera')
        os.system(str('`which chimera` '+ ' '.join(follow_opt_df_last_ite_model)))

    ### RETURN

    return ([follow_msg, follow_graph, follow_dd_y_list, classnumber_graph, follow_model, last_ite_fsc_df_list, angdist_per_class_plot])

## Callback Refine 3D
@app.callback(
    [Output(component_id='ref_follow_msg', component_property='value'),
     Output(component_id='ref_follow_graph', component_property='figure'),
     Output(component_id='ref_follow_dropdown_y', component_property='options'),
     Output(component_id='ref_follow_model', component_property='figure'),
     Output(component_id='ref_follow_model_dropdown_y', component_property='options'),
     Output(component_id='ref_follow_angdist', component_property='figure')],
    [Input(component_id='ref_job2follow', component_property='value'),
     Input(component_id='ref_follow_dropdown_y', component_property='value'),
     Input(component_id='ref_follow_model_dropdown_y', component_property='value'),
     Input(component_id='ref_display_last_ite', component_property='n_clicks'),
     Input(component_id='ref_display_chimera_last_ite', component_property='n_clicks')
     ]
)

def load_df_and_graphs(ref_job2follow, ref_follow_dd_y,ref_follow_model_dd_y,ref_display_last_ite_button_press, ref_display_last_ite_chimera_button_press):

### Follow Refine 3D

    ref_job = ref_job2follow

    if 'Refine3D' in str(ref_job):

        ref_follow_msg = 'Following Refine3D job: '+str(ref_job)

        all_model = glob.glob(os.path.join(ref_job+'run_it*_half1_model.star'))
        all_model.sort()
        stars_model1 = []
        for filename in all_model:
            optimiser_df = starfile.read(filename)['model_general']
            stars_model1.append(optimiser_df)
        ref_follow_opt_df = pd.concat(stars_model1, axis=0, ignore_index=True)
        ref_follow_dd_y_list = list(ref_follow_opt_df)
        last_ite_model = str(all_model[-1])
        last_ite_model_df = starfile.read(last_ite_model)['model_classes']
        last_ite_fsc_df = starfile.read(last_ite_model)['model_class_1']
        last_ite_fsc_df_list = list(last_ite_fsc_df)
        ref_follow_opt_df_last_ite_model = last_ite_model_df['rlnReferenceImage']
        ref_follow_opt_df_last_ite_maps = ref_follow_opt_df_last_ite_model
        resolutions = list(last_ite_fsc_df['rlnResolution'])
        number_of_iterations = len(ref_follow_opt_df[ref_follow_dd_y])-1
        angdist = starfile.read(ref_job+f'run_it{number_of_iterations:03d}_data.star')['particles'][['rlnAngleRot', 'rlnAngleTilt']]
        sampling_angle = float(starfile.read(ref_job+f'run_it{number_of_iterations:03d}_sampling.star')['sampling_general']['rlnPsiStep'])
        
    else:

        ref_follow_msg = 'Nothing to do here...'
        ref_follow_dd_y_list = ['']
        ref_follow_opt_df = pd.DataFrame()
        ref_follow_opt_df['bla'] = []
        ref_follow_classnumber_df = pd.DataFrame()
        ref_follow_dd_y = 'bla'
        ref_follow_model_dd_y = 'bla'
        last_ite_fsc_df = pd.DataFrame()
        last_ite_fsc_df['rlnResolution'] = []
        last_ite_fsc_df['bla'] = []
        last_ite_fsc_df_list = ['']
        resolutions = ['']
        number_of_iterations = 1
        angdist = pd.DataFrame()
        angdist['rlnAngleRot'] = ['']
        angdist['rlnAngleTilt'] = ['']
        sampling_angle = 1

    ref_follow_graph = plot_line(ref_follow_opt_df, range(number_of_iterations+1), ref_follow_dd_y)
    ref_follow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title='Iteration', yaxis_title=ref_follow_dd_y)

    ref_follow_model = plot_line(last_ite_fsc_df, resolutions, ref_follow_model_dd_y)
    if ref_follow_model_dd_y == 'rlnGoldStandardFsc': ref_follow_model.add_hline(y=0.143, line_dash='dash', line_color="grey")
    elif ref_follow_model_dd_y == 'rlnSsnrMap': ref_follow_model.add_hline(y=1, line_dash='dash', line_color="grey")
    ref_follow_model.update_layout(title_text=f'Iteration {number_of_iterations:03d}', title_x=0.5, xaxis_title='1/Resolution', yaxis_title = ref_follow_model_dd_y)

    angdist_plot = plot_angdist(angdist, 'rlnAngleRot', 'rlnAngleTilt', int(360//(sampling_angle*2)), [1]*len(angdist['rlnAngleRot']))
    angdist_plot.update_layout(
        xaxis = dict(tickmode= 'linear', dtick = 90),
        yaxis = dict(tickmode= 'linear', dtick = 90),
        title_text=f'AngDist',
        title_x=0.5
    )
    print(ref_follow_msg)

    # Display last ite
    ref_display_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in ref_display_last_ite_button_press_changed:
        print('displaying ' + ref_follow_opt_df_last_ite_model[0])
        os.system(str('`which relion_display` --i '+ ref_follow_opt_df_last_ite_model[0] +' --gui '))

    ref_display_last_ite_chimera_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_chimera_last_ite' in ref_display_last_ite_chimera_button_press_changed:
        print('displaying ' + ref_follow_opt_df_last_ite_maps[0] + ' in chimera')
        os.system(str('`which chimera` '+ ref_follow_opt_df_last_ite_maps[0]))

    ### RETURN

    return ([ref_follow_msg, ref_follow_graph, ref_follow_dd_y_list, ref_follow_model, last_ite_fsc_df_list, angdist_plot])


### Dash app start

if __name__ == '__main__':
    app.run_server(debug=debug_mode, dev_tools_hot_reload = False, use_reloader=True,
                   host=hostname, port=port_number)
