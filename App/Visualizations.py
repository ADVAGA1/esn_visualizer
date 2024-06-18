import tkinter as tk
from tkinter import ttk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class Visualization():
    def __init__(self):
        pass
    
    def destroy(self):
        pass

    def on_scroll(self, event):
            base_scale = 1.2
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return

            self.scale_view(event, scale_factor)

    def scale_view(self, event, scale_factor):
        xdata = event.xdata
        ydata = event.ydata

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw()

    def on_press(self, event):
        if event.button == 1:
            self.dragging = True
            self.last_press = (event.x, event.y)

    def on_release(self, event):
        self.dragging = False
        self.last_press = None

    def on_motion(self, event):
        if self.dragging and self.last_press is not None:
            dx = event.x - self.last_press[0]
            dy = event.y - self.last_press[1]

            self.last_press = (event.x, event.y)

            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()

            dx = dx / self.canvas.get_tk_widget().winfo_width() * (cur_xlim[1] - cur_xlim[0])
            dy = dy / self.canvas.get_tk_widget().winfo_height() * (cur_ylim[1] - cur_ylim[0])

            self.ax.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
            self.ax.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
            self.canvas.draw()


class PredictionVisualization(Visualization):
    def __init__(self, master, backend):
        self.master = master
        prediction, errors = backend.predict()
        n = backend.timesteps
        n_train = int(backend.tts * backend.timesteps)

        self.fig = Figure(figsize=(10,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(range(n), backend.data, label='Ground Truth')
        self.ax.plot(range(n_train+1, n), prediction, label='Predictions')
        self.ax.axvline(x = n_train+1, color = 'r', linestyle='--', linewidth=1)
        self.ax.set_xlabel(f"TimeSteps (MSE={errors['mse']:.3f}, NMSE={errors['nmse']:.3f}, RMSE={errors['rmse']:.3f}, MAE={errors['mae']:.3f}, R2={errors['rsquare']:.3f})")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Ground Truth vs Predictions")
        self.ax.legend(loc="upper right")
        xticks = self.ax.get_xticks()
        xticks = list(xticks) + [n_train+1]
        self.ax.set_xticks(xticks)
        self.ax.set_xlim(-1, n+1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, master, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.master.bind('<Configure>', lambda event: self.resize_widget(event, self.fig, self.canvas))

    def destroy(self):
        self.master.unbind('<Configure>')
        plt.close(self.fig)
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()

    def resize_widget(self, event, figure, canvas):
        canvas.resize(event)
        canvas.get_tk_widget().config(width=event.width, height=event.height)
        figure.set_size_inches(event.width / canvas.figure.dpi, event.height / canvas.figure.dpi)
        canvas.draw()

class WeightsVisualizationPyESN(Visualization):
    def __init__(self, master, backend):

        self.tmp_path = backend.tmp_path
        self.master = master

        if not os.path.isfile(self.tmp_path + "\\weights.png"):
            self.make_image(backend)
            
        self.canvas = Zoom(master, self.tmp_path + "\\weights.png")

    def make_image(self, backend):
        W_in, W_res, W_out = backend.get_weights()
        reservoir_size = backend.get_reservoir_size()

        self.fig, self.ax = plt.subplots(1,3, figsize=(reservoir_size // 2, reservoir_size // 4), gridspec_kw={'width_ratios': [1, 3, 1]})

        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

        im0 = self.ax[0].imshow(W_in[::-1], aspect='equal', extent=(0, 1, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[0].set_title("Input weights")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[0].set_yticks(major_ticks)
        self.ax[0].set_yticks(minor_ticks, minor=True)
        self.ax[0].grid(True, "both", color="black")


        im1 = self.ax[1].imshow(W_res[::-1], aspect='equal', extent=(0, reservoir_size, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[1].set_title("Reservoir weights")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[1].set_xticks(major_ticks)
        self.ax[1].set_xticks(minor_ticks, minor=True)
        self.ax[1].set_yticks(major_ticks)
        self.ax[1].set_yticks(minor_ticks, minor=True)
        # self.ax[1].tick_params(axis = "both", which="major", label_rotatio=90)
        self.ax[1].grid(True, "both", color="black")

        im2 = self.ax[2].imshow(W_out[::-1], aspect='equal', extent=(0, 1, 0, reservoir_size+1), cmap=cmap, vmin=-1, vmax=1)
        self.ax[2].set_title("Readout weights")
        major_ticks = np.arange(0, reservoir_size+1, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size+1, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[2].set_yticks(major_ticks)
        self.ax[2].set_yticks(minor_ticks, minor=True)
        self.ax[2].grid(True, "both", color="black")


        cbar = self.fig.colorbar(im2, ax=self.ax[2], orientation='vertical', location="right", pad=0.5, shrink=0.75)
        cbar.set_label('Colorbar')

        self.fig.tight_layout()
        self.fig.savefig(self.tmp_path + "\\weights.png")
        plt.close()
        print("Created weights.png")

    def destroy(self):
        self.master.unbind('<Configure>')
        self.canvas.get_tk_widget().destroy()

class WeightsVisualizationReservoirPy(Visualization):
    def __init__(self, master, backend):

        self.tmp_path = backend.tmp_path
        self.master = master

        if not os.path.isfile(self.tmp_path + "\\weights.png"):
            self.make_image(backend)
            
        self.canvas = Zoom(master, self.tmp_path + "\\weights.png")

    def make_image(self, backend):
        Win, W, reservoir_bias, Wout, readout_bias = backend.get_weights()
        reservoir_size = backend.get_reservoir_size()

        self.fig, self.ax = plt.subplots(1,5, figsize=(reservoir_size // 2, reservoir_size // 4), gridspec_kw={'width_ratios': [1, 3, 1, 1, 1]})

        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

        im0 = self.ax[0].imshow(Win[::-1], aspect='equal', extent=(0, 1, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[0].set_title("Input weights")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[0].set_yticks(major_ticks)
        self.ax[0].set_yticks(minor_ticks, minor=True)
        self.ax[0].grid(True, "both", color="black")


        im1 = self.ax[1].imshow(W[::-1], aspect='equal', extent=(0, reservoir_size, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[1].set_title("Reservoir weights")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[1].set_xticks(major_ticks)
        self.ax[1].set_xticks(minor_ticks, minor=True)
        self.ax[1].set_yticks(major_ticks)
        self.ax[1].set_yticks(minor_ticks, minor=True)
        # self.ax[1].tick_params(axis = "both", which="major", label_rotatio=90)
        self.ax[1].grid(True, "both", color="black")

        im2 = self.ax[2].imshow(reservoir_bias[::-1], extent=(0, 1, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[2].set_title("Reservoir bias")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[2].set_yticks(major_ticks)
        self.ax[2].set_yticks(minor_ticks, minor=True)
        self.ax[2].grid(True, "both", color="black")

        im3 = self.ax[3].imshow(Wout, aspect='equal', extent=(0, 1, 0, reservoir_size), cmap=cmap, vmin=-1, vmax=1)
        self.ax[3].set_title("Readout weights")
        major_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))))
        minor_ticks = np.arange(0, reservoir_size, max(1, int(np.sqrt(reservoir_size))//4))
        self.ax[3].set_yticks(major_ticks)
        self.ax[3].set_yticks(minor_ticks, minor=True)
        self.ax[3].grid(True, "both", color="black")

        im4 = self.ax[4].imshow(readout_bias, cmap=cmap, extent=(0, 1, 0, 1), vmin=-1, vmax=1)
        self.ax[4].set_title("Readout Bias")

        cbar = self.fig.colorbar(im4, ax=self.ax[4], orientation='vertical', location="right", pad=0.5, shrink=0.75)
        cbar.set_label('Colorbar')

        self.fig.tight_layout()
        self.fig.savefig(self.tmp_path + "\\weights.png")
        plt.close()
        print("Created weights.png")

    def destroy(self):
        self.master.unbind('<Configure>')
        self.canvas.get_tk_widget().destroy()

class StatesVisualization(Visualization):
    def __init__(self, master, backend, which):

        self.tmp_path = backend.tmp_path
        self.saved = False
        self.last_index = 0
        self.which = which
        self.image_dir = os.path.join(self.tmp_path, self.which)

        if not os.path.isfile(os.path.join(self.image_dir, "0.png")):
            self.image_files = backend.make_states_images(self.which)
        else:
            self.image_files = backend.get_images(self.which)

        if self.which == "train":
            self.X = backend.X_train
            self.states = backend.train_states
            self.prediction = backend.train_prediction
        else:
            self.X = backend.X_test
            self.states = backend.test_states
            self.prediction = backend.prediction

        # Load and resize images
        image = Image.open(os.path.join(self.image_dir, self.image_files[0])).resize((800, 800))
        self.tk_image = ImageTk.PhotoImage(image=image)
    
        # Create the main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill="both", expand=True)

        # Crear el frame izquierdo
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # Configurar el grid en el frame izquierdo
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=0)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Añadir la imagen grande a la izquierda
        self.image_label = ttk.Label(self.left_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.slide_var = tk.IntVar()
        self.slide_var.trace("w", lambda *args: self.update_image_and_plot(self.slide_var.get()))
        self.slider = tk.Scale(self.left_frame, from_=0, to=len(self.image_files)-1, variable=self.slide_var, tickinterval=25, orient=tk.HORIZONTAL, length=1000)
        self.slider.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Crear el frame derecho
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Configurar el grid en el frame derecho
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(2, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(2, weight=1)

        # Create the plot
        self.fig, self.ax = plt.subplots()
        time_data = np.arange(len(self.X))
        self.ax.plot(time_data, self.X, 'b-', label="Ground Truth")  # Plot the X dataset
        self.ax.plot(time_data, self.prediction, 'y-', label="Prediction")
        self.ax.legend()
        self.point, = self.ax.plot([self.last_index], [self.prediction[0]], 'ro')  # Initial point

        self.ax.set_xlim(0, len(self.X)-1)
        self.ax.set_ylim(min(min(self.prediction), min(self.X)), max(max(self.prediction), max(self.X)))
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.set_title("X Dataset over Time")

        states = 20 if self.states.shape[1] >= 20 else self.states.shape[1]

        # Create the states plot
        self.fig_states, self.ax_states = plt.subplots()
        self.ax_states.plot(self.states[:,:states], label=range(states))  # Plot the X dataset
        # self.points = []
        # for s in self.states[0]:
        #     point, = self.ax_states.plot([0], [s], 'o')  # Initial point
        #     self.points.append(point)

        self.vline = self.ax_states.axvline(x=self.last_index, color='r', linestyle='--', linewidth=1)
        self.ax_states.set_xlim(0, len(self.states)-1)
        self.ax_states.set_ylim(self.states.min(), self.states.max())
        self.ax_states.set_xlabel("Time")
        self.ax_states.set_ylabel("Value")
        self.ax_states.set_title(f"States over Time on {self.which}")
        self.ax_states.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(1.1, 1), borderaxespad=0.)

        # Create a canvas to display the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        # Create a canvas to display the plot
        self.canvas_states = FigureCanvasTkAgg(self.fig_states, master=self.right_frame)
        self.canvas_states.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.states_val = IntVar()
        self.states_val.trace("w", lambda *args: self.update_states_plot(int(self.states_val.get())))
        self.states_box = ttk.Spinbox(master=self.right_frame, textvariable=self.states_val, from_=1, to=self.states.shape[1], increment=1)
        self.states_box.grid(column=1, row=1, padx=5, pady=5)
        self.states_box.set(20)
        self.states_label = Label(master=self.right_frame, text="Neurons shown", relief='ridge')
        self.states_label.grid(column=0, row=1, padx=5, pady=5, sticky='nsew')
        
        self.save_state_button = Button(master=self.right_frame, text="Save Figure", command=self.save_state_figure)
        self.save_state_button.grid(column=2, row=1, padx=5, pady=5)

        # Initialize the first image and plot
        
        self.update_image_and_plot(self.last_index)

    def save_state_figure(self):
        image_dir = self.tmp_path + f"\\{self.which}_states.png"
        self.fig_states.savefig(image_dir)
        messagebox.showinfo("Figure saved.",f"Figure saved in {image_dir}")

    def update_image_and_plot(self, value):
        index = int(value)
        print(value)
        if not self.saved and index == 0:
            self.fig.savefig(self.tmp_path + f"\\prediction_{self.which}_states.png")
            self.fig_states.savefig(self.tmp_path + f"\\{self.which}_states.png")
            self.saved = True
        self.image = Image.open(os.path.join(self.image_dir, self.image_files[index])).resize((800, 800))
        self.tk_image = ImageTk.PhotoImage(image=self.image)
        self.image_label.config(image=self.tk_image)
        self.last_index = index
        self.update_plot(index)

    def update_plot(self, index):
        self.point.set_xdata([index])
        self.point.set_ydata([self.prediction[index]])
        self.canvas.draw()

        self.vline.set_xdata([index])
        self.canvas_states.draw()

    def update_states_plot(self, value):
        plt.close(self.fig_states)
        self.fig_states, self.ax_states = plt.subplots()
        self.ax_states.plot(self.states[:,:value], label=range(value))  # Plot the X dataset
        self.vline = self.ax_states.axvline(x=self.last_index, color='r', linestyle='--', linewidth=1)
        self.ax_states.set_xlim(0, len(self.states)-1)
        self.ax_states.set_ylim(self.states.min(), self.states.max())
        self.ax_states.set_xlabel("Time")
        self.ax_states.set_ylabel("Value")
        self.ax_states.set_title(f"States over Time on {self.which}")
        self.ax_states.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
        self.canvas_states.figure = self.fig_states
        self.canvas_states.draw()

    def destroy(self):
        plt.close(self.fig)
        plt.close(self.fig_states)
        self.main_frame.destroy()
        self.left_frame.destroy()
        self.right_frame.destroy()
        self.canvas_states.get_tk_widget().destroy()
        self.image_label.destroy()
        self.slider.destroy()
        self.states_box.destroy()
        self.states_label.destroy()


class Zoom(ttk.Frame):
    ''' Simple zoom with mouse wheel '''
    def __init__(self, mainframe, path):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        
        # Open image
        self.image = Image.open(path)
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up

        # Bind resize event to adjust image scale
        self.master.bind('<Configure>', self.resize_image)

        # Show image and plot some random test rectangles on the canvas
        self.imscale = self.master.winfo_width() / self.image.size[0]
        self.imageid = None
        self.delta = 0.75
        self.text = self.canvas.create_text(0, 0, anchor='nw', text='')
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:
            scale        *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale        /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def show_image(self):
        ''' Show image on the Canvas '''
        if self.imageid:
            self.canvas.delete(self.imageid)
            self.imageid = None
            self.canvas.imagetk = None  # delete previous image from the canvas
        width, height = self.image.size
        new_size = int(self.imscale * width), int(self.imscale * height)
        imagetk = ImageTk.PhotoImage(self.image.resize(new_size))
        # Use self.text object to set proper coordinates
        self.imageid = self.canvas.create_image(self.canvas.coords(self.text),
                                                anchor='nw', image=imagetk)
        self.canvas.lower(self.imageid)  # set it into background
        self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def get_tk_widget(self):
        return self.canvas
    
    def resize_image(self, event):
        ''' Resize image to fit window '''
        width = event.width
        self.imscale = width / float(self.image.size[0])
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))