import plotly.graph_objects as go
import networkx as nx
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
def load_data(similarity_df, similarity_matrix, partidos):
    """Load graph data from similarity matrix and other sources."""
    G = nx.Graph()
    names = similarity_df.columns

    # Add nodes to graph
    for name in names:
        G.add_node(name)

    # Add edges based on similarity matrix and threshold
    threshold = 0.70
    counter = {name: [] for name in names}
    for i in range(len(similarity_df)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                counter[names[i]].append((names[j], similarity_matrix[i][j]))

    for source, targets in counter.items():
        selected_targets = sorted(targets, key=lambda x: x[1], reverse=True)[:10]
        for target, weight in selected_targets:
            G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G, k=0.2, iterations=50, seed=14)

    # Map node groups and colors
    group_mapping = partidos.set_index('nome')['grupo'].to_dict()

# Create a graph from the similarity matrix dataframe

    group_colors = {
        'oposicao': 'blue',
        'governo': 'red',
        'centrao_mdb': 'green',
        'centrao_uniao': 'yellow',
        'outros': 'grey'  # Default color for any group not listed above
    }
    # Assign the 'group' attribute from your 'partidos' dataframe to each node in the graph
    for node in G.nodes():
        # If the node's name exists in the group mapping dictionary, assign the group value
        if node in group_mapping:
            G.nodes[node]['group'] = group_mapping[node]
        else:
            # If there's no group found for the node, assign a default value
            G.nodes[node]['group'] = 'outros'

    return G, pos, group_colors

def create_figure(G, pos, group_colors, group_chosen, edges_):
    """Generate the figure for the network graph."""
    edge_trace = []

    if edges_ == 'Edges':
        for edge in G.edges():
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos[char_1]
            x1, y1 = pos[char_2]
            trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                            mode='lines',
                            line={'width': 0.5, 'shape': 'spline'},
                            opacity=0.5)
            edge_trace.append(trace)

    # Generate node trace
    
    if group_chosen == 'all':
        nodes_chosen = G.nodes()
    else:
        nodes_chosen = [node for node in G.nodes() if G.nodes(data=True)[node]['group'] == group_chosen]
        
    node_sizes = {node: len(G.edges(node)) for node in G.nodes()}  # Node sizes by edges

    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', textposition="bottom center",
                            hoverinfo='text', marker={'size': [(node_sizes[node])*10 for node in G.nodes()], 'color': []},
                            opacity=0.8)

    for node in nodes_chosen:
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += (group_colors[G.nodes[node]['group']],)
        node_trace['text'] += tuple([f'<b>{node}</b>'])

    # Create figure layout
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        clickmode='event+select'))

    return fig

# Main app setup
app = dash.Dash(__name__)

votes_pivot = pd.read_csv('votes_pivot.csv', index_col=0)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(votes_pivot)
similarity_df = pd.DataFrame(similarity_matrix, index=votes_pivot.index, columns=votes_pivot.index)
partidos = pd.read_csv('partidos.csv')


G, pos, group_colors = load_data(similarity_df, similarity_matrix, partidos)

@app.callback(
    Output('network-graph', 'figure'),
    [Input('controls-and-radio-item', 'value'),
    Input('controls-and-dropdown', 'value')]
)
def create_figure_callback(group_chosen, edges_):
    return create_figure(G, pos, group_colors, group_chosen, edges_)

app.layout = html.Div([
    dcc.Dropdown(options=['Edges', 'No Edges'], id='controls-and-dropdown', value='Edges', style={'width': '50%'}),
    dcc.RadioItems(options=list(group_colors.keys())+['all'], value='all', id='controls-and-radio-item', labelStyle={'display': 'inline-block'}),
    dcc.Graph(id='network-graph', figure=create_figure(G, pos, group_colors, 'all', 'Edges'))
])
server = app.server
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
