from re import template
from turtle import heading
import dash # v 1.16.2
import dash_core_components as dcc # v 1.12.1
import dash_bootstrap_components as dbc # v 0.10.3
import dash_html_components as html # v 1.1.1
import pandas as pd
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output
from sklearn.exceptions import NonBLASDotWarning


external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Model Dashboard', external_stylesheets=[external_stylesheets])
server = app.server


#GLOBAL
df = pd.read_csv('D:\\iti - AI ML\\visualisation\\supermarket\\Supermarket-Dashboard\\customer_dataset.csv')
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
models = ['PCA', 'UMAP', 'AE', 'VAE']
df_average = df[features].mean()
max_val = df[features].max().max()



app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Label('Model selection'),], style={'font-size': '18px'}),

            dcc.Dropdown(
                id='crossfilter-model',
                options=[
                    {'label':'Principal Cpmonent Analysis' , 'value': 'PCA'},
                    {'label':'Uniform Manifold Approximation and projection' , 'value': 'UMAP'},
                    {'label':'Variational Autoencoder' , 'value': 'VAE'}
                ],
                value='PCA',
                clearable=False

            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            
            html.Div([
                html.Label('Feature selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.RadioItems(
                    #radio for color scheme for people who has color blindness
                    id='gradient-scheme',
                    options=[
                        {'label': 'Orange to Red' , 'value': 'OrRd'},
                        {'label': 'Vididis' , 'value': 'Viridis'},
                        {'label': 'Plasma' , 'value': 'Plasma'}
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            
            dcc.Dropdown(
                #color by... features + Region , Channel , Total_Spent
                id='crossfilter-feature',
                options=[{'label': i , 'value': i } for i in ['None'] + features + ['Region' , 'Channel' , 'Total_Spend'] ],
                value= 'None',
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
        
        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}),

    html.Div([
        dcc.Graph(
            id='scatter-plot',
            hoverData={'points': [{'customdata': 0}]}
        ),
        ], style={'width': '100%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
    
    html.Div([
        dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),




    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)


@app.callback(
    Output('scatter-plot','figure'),
    #returns elli gai men el droplist
    Input('crossfilter-feature','value'),
    Input('crossfilter-model', 'value'),
    Input('gradient-scheme', 'value')
        
)

#function mab3otlaha el value/s rli 25tartaha fi el droplist/s
def update_graph(feature, model, gradient):


    if feature == 'None' :
        #cols >> colors
        #here cols and sizes are not important to declare .. just to make it clean
        cols =None
        sizes =None
        hover_names = [ f'Customer {ix}' for ix in df.index ]
    elif feature in [ 'Region' , 'Channel' ]:
        #here cols and sizes are not important to declare .. just to make it clean
        cols = df[feature].astype(str) #too have different colors not the shades of the color
        sizes = None
        hover_names = [ f'Customer {ix}' for ix in df.index ]
    else: # feature chosen is any of the supermarket products
        cols = df[feature]
        sizes = [np.max([max_val/10 , val]) for val in df[feature].values]
        # i want to show on the hover the total spent by this customer on this product(feature)
        hover_names = []
        # appending the indeces and the
        for ix, val in zip (df.index.values , df[feature].values):
            hover_names.append(f'Customer {ix}<br>{feature} value of {val}')

    
    fig = px.scatter(
        df , 
        x=df[f'{model.lower()}_x'] , 
        y=df[f'{model.lower()}_y'] ,
        opacity=0.8 ,
        template = 'plotly_dark' ,
        #fi kol ta8iir fi el graph it will take the gradient value (color)
        color_continuous_scale= gradient,
        #insert the cols, sizes, hover_names in the plot
        hover_name = hover_names,
        color = cols,
        size = sizes
        )

    #to update the info
    #the value the plot will return when you hover over it
    fig.update_traces(customdata = df.index)

    fig.update_layout(
        height = 450,
        margin = {'l': 20 , 'b':30 , 'r': 10 , 't': 10} ,
        hovermode = 'closest', # if the user hover very close to a point it will show this point 
        template = 'plotly_dark',         
    )

    fig.update_xaxes(showticklabels = False , showgrid=False)
    fig.update_yaxes(showticklabels = False)

    return fig





@app.callback(
    Output('point-plot' , 'figure'),
    Input('scatter-plot' , 'hoverData')
)


def update_point_plot(hoverData):
    #hoverData will return as a dictionery that holds the data of the customer
    #i clicked on in the first graph
    index = hoverData['points'][0]['customdata']
    title = f'Customer {index}'
    #dataframe that contains the data only for that person (row)
    return create_point_plot(df[features].iloc[index] , title) 


def create_point_plot(df,title):
    fig = go.Figure(
        data = [
            go.Bar(name = 'Avarage' , x = features , y = df_average.values , marker_color = '#c178f6'),
            go.Bar(name = title , x = features , y = df.values , marker_color = '#89efbd'),
        ]
    )

    fig.update_layout(
        barmode = 'group',
        height = 250,
        margin = {'l': 20 , 'b':30 , 'r': 10 , 't': 10} ,
        template = 'plotly_dark'
    )
    
    #the log to avoid the small values of the bar
    fig.update_yaxes(type='log' , range=[0, 5])

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
