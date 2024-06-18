from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass, logistic_map
from reservoirpy.observables import mse, rmse, rsquare
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import networkx as nx
import copy
import re
import matplotlib.pyplot as plt
from mpire import WorkerPool
import os
from pyESN import ESN
from scipy import sparse
from matplotlib.lines import Line2D


class ReservoirpyBackend():
    def __init__(self, tmp_path):
        self.esn_params = None
        self.reservoir = None
        self.readout = None
        self.dataset = None
        self.timesteps = None
        self.tts = None
        self.created = False
        self.trained = False
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_states = None
        self.test_states = None
        self.prediction = None
        self.train_prediction = None
        self.tmp_path = tmp_path

    def reset(self):
        self.esn_params = None
        self.reservoir = None
        self.readout = None
        self.dataset = None
        self.timesteps = None
        self.tts = None
        self.created = False
        self.trained = False
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_states = None
        self.test_states = None
        self.prediction = None
        self.train_prediction = None

    def train(self):
        n_train = int(self.tts * self.timesteps)
        self.X_train = self.data[:n_train]
        self.y_train = self.data[1:n_train+1]
        self.X_test = self.data[n_train:-1]
        self.y_test = self.data[n_train + 1:]

        self.train_states = self.reservoir.run(self.X_train)
        self.readout.fit(self.train_states, self.y_train, warmup=10)
        self.train_prediction = self.readout.run(self.train_states)
        self.trained = True

        self.test_states = self.reservoir.run(self.X_test)
        self.prediction = self.readout.run(self.test_states)

    def load_data(self, filepath, timesteps, tts):
        self.data = np.load(filepath).reshape(-1,1)
        self.timesteps = timesteps if timesteps <= self.data.shape[0] else self.data.shape[0]
        self.data = self.data[:timesteps]
        self.tts = tts
        self.train()

    def esn_created(self):
        return self.created
    
    def esn_trained(self):
        return self.trained
    
    def get_test_states(self):
        return self.test_states
    
    def get_prediction(self):
        return self.prediction

    def create_ESN(self, params):
        self.esn_params = params
        self.reservoir = Reservoir(units=self.esn_params["units"],
                          lr=self.esn_params["lr"],
                          sr=self.esn_params["sr"],
                          input_scaling=self.esn_params["input"],
                          input_connectivity=self.esn_params["con"],
                          rc_connectivity=self.esn_params["con"],
                          fb_connectivity=self.esn_params["con"],
                          noise_rc=self.esn_params["noise"],
                          activation=self.esn_params["act"],
                          seed=self.esn_params["seed"])
        self.readout = Ridge(ridge=self.esn_params["learn"])
        self.created = True

    def train_ESN(self, dataset, timesteps, tts):
        self.dataset = dataset
        self.timesteps = timesteps
        self.tts = tts

        if self.dataset == "Mackey Glass":
            self.data = mackey_glass(timesteps, tau=30)
        elif self.dataset == "Logistic map":
            self.data = logistic_map(timesteps)

        self.train()

    
    def predict(self):
        
        errors = {}
        errors["mse"] = mse(self.y_test, self.prediction)
        errors["rmse"] = rmse(self.y_test, self.prediction)
        errors["mae"] = mean_absolute_error(self.y_test, self.prediction)
        errors["rsquare"] = rsquare(self.y_test, self.prediction)
        errors["nmse"] = 1 - errors["rsquare"]

        return self.prediction, errors
    
    def get_weights(self):

        W = self.reservoir.get_param('W').toarray()
        Win = self.reservoir.get_param('Win').toarray()
        reservoir_bias = self.reservoir.get_param('bias').toarray()
        Wout = self.readout.get_param('Wout')
        readout_bias = self.readout.get_param('bias')
        return Win, W, reservoir_bias, Wout, readout_bias
    
    def get_reservoir_size(self):
        return self.esn_params["units"]
    
    def get_X_test(self):
        return self.X_test
    
    def make_states_images(self, which):
        g, pos, edge_colors = self.define_graph()
        print(f"Creating {which} state figures")
        self.save_figures(g, pos, edge_colors, which, 1)
        return self.get_images(which)
    
    def get_images(self, which):
        self.image_files = [f for f in os.listdir(self.tmp_path + f"\\{which}") if f.lower().endswith(('png'))]
        sort_nicely(self.image_files)
        return self.image_files

    def define_graph(self):
        # Create reservoir graph
        reservoir_graph = nx.DiGraph()

        # Add reservoir nodes
        nodes = list(range(self.esn_params["units"]))
        reservoir_graph.add_nodes_from(nodes)
        W = self.reservoir.get_param('W')
        Win = self.reservoir.get_param('Win')
        Wout = self.readout.get_param('Wout')
        # Add reservoir edges
        for i, w in enumerate(W):
            data = w.data
            indices = w.indices
            for j, d in enumerate(data):
                color = interpolate_color_variable(d, W.data.min(), W.data.max())
                reservoir_graph.add_edge(i, int(indices[j]), weight= abs(d), color=color)
        
        # Calculate graph position
        pos = nx.spring_layout(reservoir_graph, weight="weight", iterations=100, seed=self.esn_params["seed"], k=1)
        
        # Create in and out nodes
        g_inout = nx.DiGraph()
        g_inout.add_node("out")
        g_inout.add_node("in")
        pos["out"] = copy.deepcopy(max(pos.values(), key=lambda x: x[0]))
        pos["out"][0] += 1
        pos["out"][1] = 0
        pos["in"] = copy.deepcopy(min(pos.values(), key=lambda x: x[0]))
        pos["in"][0] -= 1
        pos["in"][1] = 0
        g = nx.compose_all([reservoir_graph, g_inout])

        # Add in edges
        rows = np.repeat(np.arange(Win.shape[0]), np.diff(Win.indptr))
        data = Win.data
        for i, r in enumerate(rows):
            color = interpolate_color_variable(data[i], -1, 1)
            g.add_edge("in", r, weight=data[i], color=color)

        # Add out edges
        for i, d in enumerate(Wout.flatten()):
            color = interpolate_color_variable(d, Wout.flatten().min(), Wout.flatten().max())
            g.add_edge(i, "out", weight=d, color=color)

        edge_colors = [data["color"] for _,_,data in g.edges(data=True)]

        return g, pos, edge_colors


    def save_figures(self, graph, pos, edge_colors, which, steps):
        
        if which == "train":
            states = self.train_states
            n = self.X_train.shape[0]
        else:
            n = self.X_test.shape[0]
            states = self.test_states

        print(n)

        i_states = list(range(0, n, steps))

        # for state in i_states:
        #     save_figures_paralell((graph, pos, edge_colors, states), state)

        with WorkerPool(shared_objects=(graph, pos, edge_colors, states, self.tmp_path, which)) as pool:
            node_colors = pool.map(save_figures_paralell, i_states, len(i_states), progress_bar=True)
        # return node_colors


class pyESNBackend():
    def __init__(self, tmp_path):
        self.esn_params = None
        self.esn = None
        self.dataset = None
        self.timesteps = None
        self.tts = None
        self.created = False
        self.trained = False
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_states = None
        self.test_states = None
        self.prediction = None
        self.train_prediction = None
        self.tmp_path = tmp_path

    def reset(self):
        self.esn_params = None
        self.esn = None
        self.dataset = None
        self.timesteps = None
        self.tts = None
        self.created = False
        self.trained = False
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_states = None
        self.test_states = None
        self.prediction = None
        self.train_prediction = None

    def train(self):
        n_train = int(self.tts * self.timesteps)
        self.X_train = self.data[:n_train]
        self.y_train = self.data[1:n_train+1]
        self.X_test = self.data[n_train:-1]
        self.y_test = self.data[n_train + 1:]

        self.train_prediction = self.esn.fit(self.X_train, self.y_train)
        self.train_states = self.esn.train_states
        self.trained = True

        self.prediction = self.esn.predict(self.X_test)
        self.test_states = self.esn.test_states
        

    def load_data(self, filepath, timesteps, tts):
        self.data = np.load(filepath)
        self.timesteps = timesteps if timesteps <= self.data.shape[0] else self.data.shape[0]
        self.data = self.data[:timesteps]
        self.tts = tts
        self.train()

    def esn_created(self):
        return self.created
    
    def esn_trained(self):
        return self.trained
    
    def get_test_states(self):
        return self.test_states
    
    def get_prediction(self):
        return self.prediction

    def create_ESN(self, params):
        self.esn_params = params
        self.esn = ESN(n_inputs=1,
                       n_outputs=1,
                       n_reservoir=self.esn_params["units"],
                       spectral_radius=self.esn_params["sr"],
                       sparsity= 1 - self.esn_params["con"],
                       random_state= self.esn_params["seed"])
        self.created = True

    def train_ESN(self, dataset, timesteps, tts):
        self.dataset = dataset
        self.timesteps = timesteps
        self.tts = tts

        if self.dataset == "Mackey Glass":
            self.data = mackey_glass(timesteps, tau=30).reshape(-1,1)
        elif self.dataset == "Logistic map":
            self.data = logistic_map(timesteps).reshape(-1,1)

        self.train()

    
    def predict(self):
        
        errors = {}
        errors["mse"] = mse(self.y_test, self.prediction)
        errors["rmse"] = rmse(self.y_test, self.prediction)
        errors["mae"] = mean_absolute_error(self.y_test, self.prediction)
        errors["rsquare"] = rsquare(self.y_test, self.prediction)
        errors["nmse"] = 1 - errors["rsquare"]

        return self.prediction, errors
    
    def get_weights(self):

        W = self.esn.W
        Win = self.esn.W_in
        Wout = self.esn.W_out.reshape(-1,1)
        return Win, W, Wout
    
    def get_reservoir_size(self):
        return self.esn_params["units"]
    
    def get_X_test(self):
        return self.X_test
    
    def make_states_images(self):
        g, pos, edge_colors = self.define_graph()
        print("Creating state figures")
        self.save_figures(g, pos, edge_colors, self.test_states, 1)
        return self.get_images()
    
    def get_images(self):
        self.image_files = [f for f in os.listdir(self.tmp_path) if f.lower().endswith(('png'))]
        if "weights.png" in self.image_files:
            self.image_files.remove("weights.png")
        sort_nicely(self.image_files)
        return self.image_files

    def define_graph(self):
        # Create reservoir graph
        reservoir_graph = nx.DiGraph()

        # Add reservoir nodes
        nodes = list(range(self.esn_params["units"]))
        reservoir_graph.add_nodes_from(nodes)
        W = sparse.csr_matrix(self.esn.W)
        Win = sparse.csr_matrix(self.esn.W_in)
        Wout = self.esn.W_out
        # Add reservoir edges
        for i, w in enumerate(W):
            data = w.data
            indices = w.indices
            for j, d in enumerate(data):
                color = interpolate_color_variable(d, W.data.min(), W.data.max())
                reservoir_graph.add_edge(i, int(indices[j]), weight= abs(d), color=color)
        
        # Calculate graph position
        pos = nx.spring_layout(reservoir_graph, weight="weight", iterations=100, seed=self.esn_params["seed"], k=1)
        
        # Create in and out nodes
        g_inout = nx.DiGraph()
        g_inout.add_node("out")
        g_inout.add_node("in")
        pos["out"] = copy.deepcopy(max(pos.values(), key=lambda x: x[0]))
        pos["out"][0] += 1
        pos["out"][1] = 0
        pos["in"] = copy.deepcopy(min(pos.values(), key=lambda x: x[0]))
        pos["in"][0] -= 1
        pos["in"][1] = 0
        g = nx.compose_all([reservoir_graph, g_inout])

        # Add in edges
        rows = np.repeat(np.arange(Win.shape[0]), np.diff(Win.indptr))
        data = Win.data
        for i, r in enumerate(rows):
            color = interpolate_color_variable(data[i], -1, 1)
            g.add_edge("in", r, weight=data[i], color=color)

        # Add out edges
        for i, d in enumerate(Wout.flatten()):
            if i == len(Wout.flatten()) - 1:
                break
            color = interpolate_color_variable(d, Wout.flatten().min(), Wout.flatten().max())
            g.add_edge(i, "out", weight=d, color=color)

        edge_colors = [data["color"] for _,_,data in g.edges(data=True)]

        return g, pos, edge_colors


    def save_figures(self, graph, pos, edge_colors, states, steps):
        
        n = self.X_test.shape[0]
        print(n)

        i_states = list(range(0, n, steps))

        # for state in i_states:
        #     save_figures_paralell((graph, pos, edge_colors, states), state)

        with WorkerPool(shared_objects=(graph, pos, edge_colors, states, self.tmp_path)) as pool:
            node_colors = pool.map(save_figures_paralell, i_states, len(i_states), progress_bar=True)
        # return node_colors


def sort_nicely( l ):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def interpolate_color_variable(weight, min_val, max_val):
    # # Ensure weight is within the range
    # if weight < min_val:
    #     weight = min_val
    # if weight > max_val:
    #     weight = max_val

    # # Calculate the normalized weight in the range [0, 1]
    # # Shift the range to [0, max_val - min_val]
    # shifted_weight = weight - min_val
    # total_range = max_val - min_val
    
    # normalized_weight = shifted_weight / total_range
    
    # # Calculate midpoint for interpolation
    # midpoint = abs(min_val) / total_range

    # if normalized_weight < midpoint:
    #     # Interpolate between red and white
    #     ratio = normalized_weight / midpoint
    #     red = 1
    #     green = ratio * 1
    #     blue = ratio * 1
    # else:
    #     # Interpolate between white and green
    #     ratio = (normalized_weight - midpoint) / (1 - midpoint)
    #     red = (1 - ratio) * 1
    #     green = 1
    #     blue = (1 - ratio) * 1

    if weight > 0:
        return (0,1,0)
    elif weight == 0:
        return (0,0,1)
    else:
        return (1,0,0)

    return (red, green, blue)

def save_figures_paralell(shared_objects, state):

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive value',markerfacecolor='g', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Negative value',markerfacecolor='r', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='No value',markerfacecolor='b', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='In-out nodes',markerfacecolor='k', markersize=15)        
    ]

    graph, pos, edge_colors, states, tmp_path, which = shared_objects
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_title(f"State nº{state} graph")
    node_colors = [interpolate_color_variable(e,-1,1) for e in states[state]] + [(0,0,0),(0,0,0)]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    _ = ax.axis("off")
    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig(f"{tmp_path}\\{which}\\{state}.png")
    plt.close()
    return node_colors