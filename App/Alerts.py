from tkinter import ttk, Toplevel, Label
from PIL import Image, ImageTk

class Alert():
    def __init__(self, master, title, text):
        self.window = Toplevel(master)
        self.window.title(title)
        icon_path = "App\\logo.jfif"
        load = Image.open(icon_path)
        render = ImageTk.PhotoImage(load)
        self.window.iconphoto(False, render)

        self.label = ttk.Label(self.window, text=text)
        self.label.grid(column=0, row=0, padx=5, pady=5)

        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        self.window.protocol('WM_DELETE_WINDOW', self.window.destroy)
        
    def create(self):
        self.window.focus_force()
        self.window.mainloop()

class CreationAlert(Alert):
    def __init__(self, master):
        title = "ESN must be created"
        text = "ESN must be created before training"
        super().__init__(master, title, text)
        self.create()

class TrainAlert(Alert):
    def __init__(self, master):
        title = "ESN already trained"
        text = "ESN has been already trained."
        super().__init__(master, title, text)
        self.create()

class VisualizationAlert(Alert):
    def __init__(self, master):
        title = "ESN must be created and trained"
        text = "ESN must be created and trained before visualization"
        super().__init__(master, title, text)
        self.create()

class FrameAlert():
    def __init__(self, array, which, backend=None):
        if which == "create" and backend == "reservoirpy":
            check_dict = {
                0: "Units: Value must be between 10 and 300",
                1: "Leaking rate: Value must be between 0 and 1",
                2: "Spectal radius: Value must be greater than 0",
                3: "Connectivity: Value must be between 0 and 1",
                4: "Learning rate: Value must be greater than 0"
            }
        elif which == "create" and backend == "pyesn":
            check_dict = {
                0: "Units: Value must be between 10 and 300",
                1: "Spectal radius: Value must be greater than 0",
                2: "Connectivity: Value must be between 0 and 1",
                3: "Noise: Value must be greater than 0",
                4: "Input scaling: Value must be greater than 0"
            }
        elif which == "train":
            check_dict = {
                0: "Timesteps: Value must be positive",
                1: "Train/Test split: Value must be between 0 and 1"
            }

        alert = Toplevel()
        icon_path = "App\\logo.jfif"
        load = Image.open(icon_path)
        render = ImageTk.PhotoImage(load)
        alert.iconphoto(False, render)
        alert.title("Alert")

        label = Label(alert, text="The following values are wrong:")
        label.grid(column=0, row=0, padx=5, pady=5)
        row = 1
        for i, value in enumerate(array):
            if not value:
                label_i = Label(alert, text=check_dict[i])
                label_i.grid(column=0, row=row, padx=5, pady=5)
                row += 1

        alert.protocol('WM_DELETE_WINDOW', alert.destroy)

        alert.focus_force()
        alert.mainloop()
