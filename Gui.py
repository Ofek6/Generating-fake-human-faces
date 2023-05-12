import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


class GUI:
    def __init__(self, app):
        self.app = app
        self.create_window()


    def create_window(self):
        self.window = tk.Tk()
        self.window.geometry("1920x1080")
        self.window.title("generating fake human faces screen")
        self.window.configure(background="black")

        self.image_label = tk.Label(self.window)
        self.image_label.place(x=100, y=75)

        Button(self.window, text="Hi there, you are welcome to press the buttons and see what's happening",
               width=60).place(x=450, y=10)

        self.v = tk.IntVar()
        self.v.set(101)

        buttons_options = [("personal details", 101), ("network architecture", 102), ("losses scale ", 103),
                           ("0    ", 104), ("20  ", 105), ("40  ", 106), ("60  ", 107), ("80  ", 108), ("100", 109)]
        tk.Label(self.window, text='show the image of:', justify=tk.LEFT, padx=20).place(x=800, y=174)
        tk.Label(self.window, text="""choose number of epoch:""", justify=tk.LEFT, padx=20).place(x=800, y=303)
        for i, (button, val) in enumerate(buttons_options):
            if i > 2:
                tk.Radiobutton(self.window, text=button, padx=20, variable=self.v, command=self.app.show_choice,
                               value=val).place(x=800, y=196 + 52 + (i * 26))
            else:
                tk.Radiobutton(self.window, text=button, padx=20, variable=self.v, command=self.app.show_choice,
                               value=val).place(x=800, y=196 + (i * 26))
        Button(self.window, text="Exit", width=11, command=self.close_window).place(x=800, y=246 + (i * 26) + 40)

    def close_window(self):
        self.window.destroy()
        exit()

