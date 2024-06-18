from Application import Application
import tkinter as tk

WIDTH = 1024
HEIGHT = 720


class EntryPoint():
    def __init__(self):

        self.window = tk.Tk()
        self.window.title("Backend")

        label = tk.Label(master=self.window , text="Which backend do you want to use?")
        label.grid(column=0,row=0,columnspan=2, padx=5, pady=5)

        button_pyesn = tk.Button(master=self.window , text="PyESN", command=self.pyesn)
        button_pyesn.grid(column=0, row=1, padx=5, pady=5)

        button_pyesn = tk.Button(master=self.window , text="ReservoirPy", command=self.reservoirpy)
        button_pyesn.grid(column=1, row=1, padx=5, pady=5)

        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

    def run(self):
        self.window.focus_get()
        self.window.mainloop()

    def pyesn(self):
        self.backend = "pyESN"
        self.start_app()

    def reservoirpy(self):
        self.backend = "reservoirpy"
        self.start_app()

    def start_app(self):
        print(self.backend)
        self.window.destroy()
        app = Application(WIDTH, HEIGHT, self.backend)
        app.run()


if __name__ == "__main__":
    entry = EntryPoint()
    entry.run()