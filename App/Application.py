from tkinter import *
import tkinter as tk
from tkinter import ttk
from Frames import *
from Backend import *
from Alerts import *
import os
import shutil
from PIL import Image, ImageTk


class Application():
    def __init__(self, width, height, backend):
        self.width = width
        self.height = height
        self.root = Tk()
        icon_path = "App\\logo.jfif"
        load = Image.open(icon_path)
        render = ImageTk.PhotoImage(load)
        self.root.iconphoto(False, render)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.minsize(640, 360)
        self.root.title("ESN Visualization App")
        self.tmp_path = ".\\tmp"
        if backend == "pyESN":
            self.backend = pyESNBackend(self.tmp_path)
        else:
            self.backend = ReservoirpyBackend(self.tmp_path)
        

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        network_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Network options", menu=network_menu)
        network_menu.add_command(label="Create Network", command=self.create_network)
        
        train_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Training", menu=train_menu)
        train_menu.add_command(label="Train Network", command=self.train_network)

        vis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualization", menu=vis_menu)
        vis_menu.add_command(label="Visualize Prediction", command=self.visualize_prediction)
        vis_menu.add_command(label="Visualize Weights", command=self.visualize_weights)
        vis_menu.add_command(label="Visualize Network Train States", command=self.visualize_train_states)
        vis_menu.add_command(label="Visualize Network Test States", command=self.visualize_test_states)

        self.vis_frame = VisualizationFrame(self.root, self.backend, self.tmp_path)
        self.vis_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.root.protocol('WM_DELETE_WINDOW', self.destroy)

        self.create_tmp()

    def destroy(self):
        self.vis_frame.destroy()
        self.delete_tmp()
        print("TMP deleted")
        self.root.destroy()

    def create_tmp(self):
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
            os.mkdir(self.tmp_path + "\\train")
            os.mkdir(self.tmp_path + "\\test")

    def delete_tmp(self):
        if not os.path.exists(self.tmp_path):
            return
        shutil.rmtree(self.tmp_path)
    
    def clean_tmp(self):
        self.delete_tmp()
        self.create_tmp()

    def run(self):
        self.root.focus_set()
        self.root.mainloop()

    def create_network(self):
        self.backend.reset()
        self.clean_tmp()
        if isinstance(self.backend, pyESNBackend):
            create_window = CreateWindowPyESN(self.root, self.backend)
        elif isinstance(self.backend, ReservoirpyBackend):
            create_window = CreateWindowReservoirPy(self.root, self.backend)
        if create_window.esn_created():
            self.vis_frame.update_state("created")

    def train_network(self):
        if not self.backend.esn_created():
            self.raise_alert("creation")
            return
        if self.backend.esn_trained():
            self.raise_alert("training")
        
        train_window = TrainWindow(self.root, self.backend)
        if train_window.esn_trained():
            self.vis_frame.update_state("trained")

    def visualize_prediction(self):
        if not self.backend.esn_created() or not self.backend.esn_trained():
            self.raise_alert("visualization")
            return
        self.vis_frame.update_state("visualize_prediction")
        

    def visualize_weights(self):
        if not self.backend.esn_created() or not self.backend.esn_trained():
            self.raise_alert("visualization")
            return
        self.vis_frame.update_state("visualize_weights")
            
    def visualize_train_states(self):
        if not self.backend.esn_created() or not self.backend.esn_trained():
            self.raise_alert("visualization")
            return
        self.vis_frame.update_state("visualize_train_states")

    def visualize_test_states(self):
        if not self.backend.esn_created() or not self.backend.esn_trained():
            self.raise_alert("visualization")
            return
        self.vis_frame.update_state("visualize_test_states")
        
    def raise_alert(self, which):
        if which == "creation":
            CreationAlert(self.root)
        elif which == "training":
            TrainAlert(self.root)
        else:
            VisualizationAlert(self.root)

if __name__ == "__main__":
    WIDTH = 1024
    HEIGHT = 720

    app = Application(WIDTH, HEIGHT, "reservoirpy")
    app.run()