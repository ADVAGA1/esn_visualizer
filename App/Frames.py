import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog, messagebox
from Visualizations import *
from Alerts import FrameAlert
from Backend import *
from PIL import Image, ImageTk

class VisualizationFrame(tk.Frame):
    def __init__(self, parent, backend, tmp_path, state = None, *args, **kwargs):
        super().__init__(master=parent, *args, **kwargs)

        self.state = state
        self.backend = backend
        self.tmp_path = tmp_path
        self.visualization = None
        
        self.label = tk.Label(self, text="Network not loaded. Please create or load a network.", font=("Verdana", 18))
        self.label.pack(expand=True)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

    def check_components(self, state):
        if state != "created":
            self.label.pack_forget()

        if self.visualization:
            self.visualization.destroy()

    def destroy(self):
        if self.label:
            self.label.destroy()
        if self.visualization:
            self.visualization.destroy()
        super().destroy()

    def update_state(self, state):
        self.state = state
        if self.state == "created":
            self.check_components(self.state)
            self.label.pack(expand=True)
            self.label.config(text="Network loaded. Please train the network.", font=("Verdana", 18))
        
        if self.state == "trained":
            self.label.config(text="Network trained. Select a visualization.", font=("Verdana", 18))
        
        if self.state == "visualize_prediction":
            self.check_components(self.state)
            self.visualize_prediction()

        if self.state == "visualize_weights":
            self.check_components(self.state)
            self.visualize_weights()

        if self.state == "visualize_train_states":
            self.check_components(self.state)
            self.visualize_states("train")

        if self.state == "visualize_test_states":
            self.check_components(self.state)
            self.visualize_states("test")
    
    def visualize_prediction(self):
        self.visualization = PredictionVisualization(master=self, backend=self.backend)
        
    def visualize_weights(self):
        if isinstance(self.backend, pyESNBackend):
            self.visualization = WeightsVisualizationPyESN(master=self, backend=self.backend)
        if isinstance(self.backend, ReservoirpyBackend):
            self.visualization = WeightsVisualizationReservoirPy(master=self, backend=self.backend)

    def visualize_states(self, which):
        self.visualization = StatesVisualization(master=self, backend=self.backend, which=which)
      
class CreateWindow():
    def __init__(self, parent, backend):
        self.window = tk.Toplevel(parent)
        icon_path = "App\\logo.jfif"
        load = Image.open(icon_path)
        render = ImageTk.PhotoImage(load)
        self.window.iconphoto(False, render)

        self.window.title("Create ESN")
        self.backend = backend

    def create(self):
        self.window.focus_get()
        self.window.mainloop()
        
class CreateWindowPyESN(CreateWindow):
    def __init__(self, master, backend):
        super().__init__(master, backend)

        self.check_array = [True] * 5

        title = tk.Label(self.window, text="Create ESN")
        title.grid(column=0, columnspan=2, row=0, padx=5, pady=5)

        reservoir_label = tk.Label(self.window, text="Parameters")
        reservoir_label.grid(column=0, columnspan=2, row=1, padx=5, pady=5)

        # Reservoir params
        # Units
        self.units_val = StringVar()
        units_label = tk.Label(self.window, text="Units")
        units_label.grid(column=0, row=2, padx=5, pady=5)
        units_box = ttk.Spinbox(self.window, from_=10.0, to=300.0, textvariable=self.units_val, increment=10)
        units_box.set(30)
        units_box.grid(column=1, row=2, padx=5, pady=5)

        # Spectral radius
        self.sr_val = StringVar()
        sr_label = tk.Label(self.window, text="Spectral radius")
        sr_label.grid(column=0, row=3, padx=5, pady=5)
        sr_box = ttk.Spinbox(self.window, from_=0.1, to=10, textvariable=self.sr_val, increment=0.25)
        sr_box.set(1.25)
        sr_box.grid(column=1, row=3, padx=5, pady=5)

        # Connectivity
        self.con_val = StringVar()
        con_label = tk.Label(self.window, text="Connectivity")
        con_label.grid(column=0, row=4, padx=5, pady=5)
        con_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.con_val, increment=0.1)
        con_box.set(0.1)
        con_box.grid(column=1, row=4, padx=5, pady=5)

        # Noise
        self.noise_val = StringVar()
        noise_label = tk.Label(self.window, text="Noise")
        noise_label.grid(column=0, row=5, padx=5, pady=5)
        noise_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.noise_val, increment=0.001)
        noise_box.set(0.001)
        noise_box.grid(column=1, row=5, padx=5, pady=5)

        # Input scaling
        self.input_val = StringVar()
        input_label = tk.Label(self.window, text="Input scaling")
        input_label.grid(column=0, row=6, padx=5, pady=5)
        input_box = ttk.Spinbox(self.window, from_=0, to=5, textvariable=self.input_val, increment=0.2)
        input_box.set(1)
        input_box.grid(column=1, row=6, padx=5, pady=5)

        # Teacher forcing
        self.teacher_val = BooleanVar()
        teacher_label = tk.Label(self.window, text="Teacher forcing")
        teacher_label.grid(column=0, row=7, padx=5, pady=5)
        teacher_box = ttk.Checkbutton(self.window, variable=self.teacher_val)
        teacher_box.grid(column=1, row=7, padx=5, pady=5)

        # Seed
        self.seed_val = StringVar()
        seed_label = tk.Label(self.window, text="Seed")
        seed_label.grid(column=0, row=8, padx=5, pady=5)
        seed_box = ttk.Spinbox(self.window, from_=0, to=100, textvariable=self.seed_val, increment=1)
        seed_box.set(33)
        seed_box.grid(column=1, row=8, padx=5, pady=5)

        # Create button
        create_button = tk.Button(self.window, text="Create ESN", command=self.create_command)
        create_button.grid(column=2, row=9, padx=5, pady=5)

        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.columnconfigure(2, weight=1)
        self.window.columnconfigure(3, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.window.rowconfigure(3, weight=1)
        self.window.rowconfigure(4, weight=1)
        self.window.rowconfigure(5, weight=1)
        self.window.rowconfigure(6, weight=1)
        self.window.rowconfigure(7, weight=1)
        self.window.rowconfigure(8, weight=1)
        self.window.rowconfigure(9, weight=1)

        self.window.protocol('WM_DELETE_WINDOW', self.window.destroy)


    def create_command(self):

        if not self.check_all():
            FrameAlert(self.check_array, "create", "pyesn")
            return
        
        ESN_params = {}
        ESN_params["units"] = int(self.units_val.get())
        ESN_params["sr"] = float(self.sr_val.get())
        ESN_params["con"] = float(self.con_val.get())
        ESN_params["noise"] = float(self.noise_val.get())
        ESN_params["input"] = float(self.input_val.get())
        ESN_params["teacher"] = self.teacher_val.get()
        ESN_params["seed"] = int(self.seed_val.get())

        self.backend.create_ESN(ESN_params)

        messagebox.showinfo("ESN Created", f'ESN created successfully!\nUnits: {ESN_params["units"]}\nSpectral radius: {ESN_params["sr"]}\nConnectivity: {ESN_params["con"]}\nNoise: {ESN_params["noise"]}\nInput scaling: {ESN_params["input"]}\nTeacher forcing: {ESN_params["teacher"]}\nSeed: {ESN_params["seed"]}')

        self.window.quit()
        self.window.destroy()

    def check_all(self):
        self.check_units()
        self.check_sr()
        self.check_con()
        self.check_noise()
        self.check_input()
        return not any(not x for x in self.check_array)

    def check_units(self):
        value = int(self.units_val.get())
        if value < 10 or value > 300:
            self.check_array[0] = False
        else:
            self.check_array[0] = True

    def check_noise(self):
        value = float(self.noise_val.get())
        if value < 0:
            self.check_array[1] = False
        else:
            self.check_array[1] = True

    def check_sr(self):
        value = float(self.sr_val.get())
        if value <= 0:
            self.check_array[2] = False
        else:
            self.check_array[2] = True

    def check_con(self):
        value = float(self.con_val.get())
        if value <= 0 or value > 1:
            self.check_array[3] = False
        else:
            self.check_array[3] = True

    def check_input(self):
        value = float(self.input_val.get())
        if value <= 0:
            self.check_array[4] = False
        else:
            self.check_array[4] = True

class CreateWindowReservoirPy(CreateWindow):
    def __init__(self, master, backend):
        super().__init__(master, backend)

        self.check_array = [True] * 7
        self.created = False

        title = tk.Label(self.window, text="Create ESN", relief='ridge')
        title.grid(column=0, columnspan=4, row=0, padx=5, pady=5, sticky='nsew')

        reservoir_label = tk.Label(self.window, text="Reservoir", relief='ridge')
        reservoir_label.grid(column=0, columnspan=2, row=1, padx=5, pady=5, sticky='nsew')
        ridge_label = tk.Label(self.window, text="Ridge", relief='ridge')
        ridge_label.grid(column=2, columnspan=2, row=1, padx=5, pady=5, sticky='nsew')

        # Reservoir params
        # Units
        self.units_val = StringVar()
        units_label = tk.Label(self.window, text="Units")
        units_label.grid(column=0, row=2, padx=5, pady=5)
        units_box = ttk.Spinbox(self.window, from_=10.0, to=300.0, textvariable=self.units_val, increment=10)
        units_box.set(30)
        units_box.grid(column=1, row=2, padx=5, pady=5)

        # Leaky rate
        self.lr_val = StringVar()
        lr_label = tk.Label(self.window, text="Leaky rate")
        lr_label.grid(column=0, row=3, padx=5, pady=5)
        lr_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.lr_val, increment=0.1)
        lr_box.set(0.3)
        lr_box.grid(column=1, row=3, padx=5, pady=5)

        # Spectral radius
        self.sr_val = StringVar()
        sr_label = tk.Label(self.window, text="Spectral radius")
        sr_label.grid(column=0, row=4, padx=5, pady=5)
        sr_box = ttk.Spinbox(self.window, from_=0.1, to=10, textvariable=self.sr_val, increment=0.25)
        sr_box.set(1.25)
        sr_box.grid(column=1, row=4, padx=5, pady=5)

        # Input scaling
        self.input_val = StringVar()
        input_label = tk.Label(self.window, text="Input scaling")
        input_label.grid(column=0, row=5, padx=5, pady=5)
        input_box = ttk.Spinbox(self.window, from_=0, to=2, textvariable=self.input_val, increment=0.1)
        input_box.set(1)
        input_box.grid(column=1, row=5, padx=5, pady=5)

        # Connectivity
        self.con_val = StringVar()
        con_label = tk.Label(self.window, text="Connectivity")
        con_label.grid(column=0, row=6, padx=5, pady=5)
        con_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.con_val, increment=0.1)
        con_box.set(0.1)
        con_box.grid(column=1, row=6, padx=5, pady=5)

        # Noise
        self.noise_val = StringVar()
        noise_label = tk.Label(self.window, text="Noise")
        noise_label.grid(column=0, row=7, padx=5, pady=5)
        noise_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.noise_val, increment=0.001)
        noise_box.set(0)
        noise_box.grid(column=1, row=7, padx=5, pady=5)

        # Activation
        self.act_val = StringVar()
        act_label = tk.Label(self.window, text="Units")
        act_label.grid(column=0, row=8, padx=5, pady=5)
        act_box = ttk.Combobox(self.window, values=["tanh", "relu", "sigmoid", "softmax"], textvariable=self.act_val)
        act_box.set("tanh")
        act_box.grid(column=1, row=8, padx=5, pady=5)

        # Seed
        self.seed_val = StringVar()
        seed_label = tk.Label(self.window, text="Seed")
        seed_label.grid(column=0, row=9, padx=5, pady=5)
        seed_box = ttk.Spinbox(self.window, from_=0, to=100, textvariable=self.seed_val, increment=1)
        seed_box.set(33)
        seed_box.grid(column=1, row=9, padx=5, pady=5)

        # Ridge params
        # Learning rate
        self.learn_val = StringVar()
        learn_label = tk.Label(self.window, text="Learning rate")
        learn_label.grid(column=2, row=2, padx=5, pady=5)
        learn_box = ttk.Spinbox(self.window, from_=0.000001, to=1, textvariable=self.learn_val, increment=0.00001)
        learn_box.set(0.00001)
        learn_box.grid(column=3, row=2, padx=5, pady=5)

        # Create button
        create_button = tk.Button(self.window, text="Create ESN", command=self.create_command)
        create_button.grid(column=3, row=10, padx=5, pady=5)

        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.columnconfigure(2, weight=1)
        self.window.columnconfigure(3, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.window.rowconfigure(3, weight=1)
        self.window.rowconfigure(4, weight=1)
        self.window.rowconfigure(5, weight=1)
        self.window.rowconfigure(6, weight=1)
        self.window.rowconfigure(7, weight=1)
        self.window.rowconfigure(8, weight=1)
        self.window.rowconfigure(9, weight=1)
        self.window.rowconfigure(10, weight=1)

        self.window.protocol('WM_DELETE_WINDOW', self.destroy)

        self.window.focus_force()
        self.window.mainloop()

    def destroy(self):
        self.window.quit()
        self.window.destroy()

    def esn_created(self):
        return self.created

    def create_command(self):
        if not self.check_all():
            FrameAlert(self.check_array, "create", "reservoirpy")
            return
            
        ESN_params = {}
        ESN_params["units"] = int(self.units_val.get())
        ESN_params["lr"] = float(self.lr_val.get())
        ESN_params["sr"] = float(self.sr_val.get())
        ESN_params["input"] = float(self.input_val.get())
        ESN_params["con"] = float(self.con_val.get())
        ESN_params["noise"] = float(self.noise_val.get())
        ESN_params["act"] = self.act_val.get()
        ESN_params["seed"] = int(self.seed_val.get())
        ESN_params["learn"] = float(self.learn_val.get())

        self.backend.create_ESN(ESN_params)

        messagebox.showinfo("ESN Created", f'ESN created successfully!\nUnits: {ESN_params["units"]}\nLeaking rate: {ESN_params["lr"]}\nSpectral radius: {ESN_params["sr"]}\Input scaling: {ESN_params["input"]}\nConnectivity: {ESN_params["con"]}\nNoise: {ESN_params["noise"]}\nActivation: {ESN_params["act"]}\nLearning rate: {ESN_params["learn"]}\nSeed: {ESN_params["seed"]}')

        print("ESN created")
        self.created = True

        self.destroy()

    def check_all(self):
        self.check_units()
        self.check_lr()
        self.check_sr()
        self.check_con()
        self.check_learn()
        return not any(not x for x in self.check_array)

    def check_units(self):
        value = int(self.units_val.get())
        if value < 10 or value > 300:
            self.check_array[0] = False
        else:
            self.check_array[0] = True

    def check_lr(self):
        value = float(self.lr_val.get())
        if value < 0 or value > 1:
            self.check_array[1] = False
        else:
            self.check_array[1] = True

    def check_sr(self):
        value = float(self.sr_val.get())
        if value <= 0:
            self.check_array[2] = False
        else:
            self.check_array[2] = True

    def check_con(self):
        value = float(self.con_val.get())
        if value <= 0 or value > 1:
            self.check_array[3] = False
        else:
            self.check_array[3] = True

    def check_learn(self):
        value = float(self.learn_val.get())
        if value <= 0:
            self.check_array[4] = False
        else:
            self.check_array[4] = True

class TrainWindow():
    def __init__(self, parent, backend):
        self.window = tk.Toplevel(parent)
        icon_path = "App\\logo.jfif"
        load = Image.open(icon_path)
        render = ImageTk.PhotoImage(load)
        self.window.iconphoto(False, render)
        self.backend = backend
        self.check_array = [True] * 2
        self.trained = False

        self.window.title("Training the ESN")

        label = ttk.Label(self.window, text="Which dataset do you wanna train?",relief='ridge')
        label.grid(column=0, row=0, columnspan=2, padx=5, pady=5,sticky='nsew')

        self.dataset_val = StringVar()
        dataset_label = ttk.Label(self.window, text="Dataset")
        dataset_label.grid(column=0, row=1, padx=5, pady=5)
        dataset_box = ttk.Combobox(self.window, values=["Mackey Glass", "Logistic map"], textvariable=self.dataset_val)
        dataset_box.set("Mackey Glass")
        dataset_box.grid(column=1, row=1, padx=5, pady=5)

        self.timesteps_val = StringVar()
        timesteps_label = ttk.Label(self.window, text="Timesteps")
        timesteps_label.grid(column=0, row=2, padx=5, pady=5)
        timesteps_box = ttk.Spinbox(self.window, from_=1, to=10_000, textvariable=self.timesteps_val, increment=100)
        timesteps_box.set(500)
        timesteps_box.grid(column=1, row=2, padx=5, pady=5)

        self.train_test_split_val = StringVar()
        tts_label = ttk.Label(self.window, text="Train/Test Split")
        tts_label.grid(column=0, row=3, padx=5, pady=5)
        tts_box = ttk.Spinbox(self.window, from_=0, to=1, textvariable=self.train_test_split_val, increment=0.1)
        tts_box.set(0.7)
        tts_box.grid(column=1, row=3, padx=5, pady=5)

        load_button = ttk.Button(self.window, text="Load Dataset", command=self.load_dataset)
        load_button.grid(column=0, row=4, padx=5, pady=5)

        train_button = ttk.Button(self.window, text="Train", command=self.train)
        train_button.grid(column=1, row=4, padx=5, pady=5)

        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.window.rowconfigure(3, weight=1)
        self.window.rowconfigure(4, weight=1)

        self.window.protocol('WM_DELETE_WINDOW', self.destroy)

        self.window.focus_force()
        self.window.mainloop()

    def destroy(self):
        self.window.quit()
        self.window.destroy()

    def esn_trained(self):
        return self.trained

    def train(self):

        if not self.check_all():
            FrameAlert(self.check_array, "train")
            return

        dataset = self.dataset_val.get()
        timesteps = int(self.timesteps_val.get())
        tts = float(self.train_test_split_val.get())


        self.backend.train_ESN(dataset, timesteps, tts)

        messagebox.showinfo("ESN Trained", f"ESN trained successfully!\nDataset: {dataset}\nTimesteps: {timesteps}\nTrain/Test Split: {tts}")

        print("ESN trained")
        self.trained = True

        self.destroy()

    def load_dataset(self):

        if not self.check_all():
            FrameAlert(self.check_array, "train")
            return

        filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not filepath:
            pass

        timesteps = int(self.timesteps_val.get())
        tts = float(self.train_test_split_val.get())
        self.backend.load_data(filepath, timesteps, tts)

        messagebox.showinfo("ESN Trained", f"ESN trained successfully!\nDataset: Own dataset\nTimesteps: {timesteps}\nTrain/Test Split: {tts}")

        self.window.quit()
        self.window.destroy()


    def check_all(self):
        self.check_timesteps()
        self.check_tts()
        return not any(not x for x in self.check_array)

    def check_timesteps(self):
        value = int(self.timesteps_val.get())
        if value <= 0:
            self.check_array[0] = False
        else:
            self.check_array[0] = True

    def check_tts(self):
        value = float(self.train_test_split_val.get())
        if value < 0 or value > 1:
            self.check_array[1] = False
        else:
            self.check_array[1] = True
