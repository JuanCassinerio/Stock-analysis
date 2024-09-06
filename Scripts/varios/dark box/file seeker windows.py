import pandas as pd
from tkinter import Tk, filedialog
from pathlib import Path
# Initialize the Tkinter root

# Perform any actions you need here
cwd = Path.cwd()
parent = cwd.parent.absolute()
realized_path = cwd / 'Scripts' / 'varios' / 'dark box'
root = Tk()
root.withdraw()

realized_file = filedialog.askopenfilename(
    initialdir=realized_path)
# Immediately destroy the window
root.destroy()

df = pd.read_csv(realized_file, sep=';')



