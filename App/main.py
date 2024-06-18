from EntryPoint import Application

WIDTH = 1024
HEIGHT = 720
BACKEND = "reservoirpy"

app = Application(width=WIDTH, height=HEIGHT, backend=BACKEND)
app.run()