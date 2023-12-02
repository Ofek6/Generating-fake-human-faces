import random
import tkinter as tk
from threading import Thread
import time
from tkinter import messagebox


class MainApplication(tk.Tk):


    def __init__(self):
        """

        """

        tk.Tk.__init__(self)
        self.title("A cool program")
        self.configure(bg="black")
        self.frames = {}

        # building the frames and inserting them to a dict
        for F in (LoginFrame, SimonFrame):
            frame = F(self)
            self.frames[F] = frame

        self.show_frame(LoginFrame)


    def show_frame(self, frame):

        # forgetting all the frames
        for F in (LoginFrame, SimonFrame):
            self.frames[F].place_forget()

        frame = self.frames[frame]
        frame.place(relheight=1, relwidth=1)


    def exit_program(self):
        self.destroy()
        exit()


class LoginFrame(tk.Frame):


    def __init__(self, master):
        """

        :param master:
        """

        tk.Frame.__init__(self, master)

        self.label_username = tk.Label(self, text="Username:")
        self.label_password = tk.Label(self, text="Password:")
        self.label_welcome = tk.Label(self, text="Welcome to my game, enter username and password for login")

        self.entry_username = tk.Entry(self)
        self.entry_password = tk.Entry(self, show="*")
        self.button_login = tk.Button(self, text="Login", command=self.login)

        self.label_welcome.place(x=210, y=0)
        self.label_username.place(x=230, y=30)
        self.label_password.place(x=230, y=50)
        self.entry_username.place(x=295, y=32)
        self.entry_password.place(x=295, y=52)
        self.button_login.place(x=330, y=80)


    def login(self):
        """

        :return:
        """

        username = self.entry_username.get()
        password = self.entry_password.get()

        # authentication
        if username == "user" and password == "password":
            self.master.show_frame(SimonFrame)

        else:
            messagebox.showerror("Login Failed", "Invalid username or password")


class SimonFrame(tk.Frame):


    def __init__(self, master):
        """

        :param master:
        """

        tk.Frame.__init__(self, master)
        self.configure(bg="black")
        self.create_frame()


    def create_frame(self):
        """

        :return:
        """
        self.go_back = tk.Button(self, text="Go Back", command=lambda: self.master.show_frame(LoginFrame)).place(x=0, y=0)
        self.exit = tk.Button(self, text="Exit", width=5, command=lambda: self.master.exit_program()).place(x=656,y=0)
        self.button_start = tk.Button(self, text="Click here to Start the Game", command=self.start_game)

        self.buttons = {
            "green": tk.Button(self, width=30, height=15, bg="green", command=lambda: self.add_color("green")),
            "blue": tk.Button(self, width=30, height=15, bg="blue", command=lambda: self.add_color("blue")),
            "red": tk.Button(self, width=30, height=15, bg="red", command=lambda: self.add_color("red")),
            "yellow": tk.Button(self, width=30, height=15, bg="yellow", command=lambda: self.add_color("yellow"))
        }

        self.button_start.place(x=290, y=0)
        self.buttons["green"].place(x=150, y=50)
        self.buttons["blue"].place(x=370, y=50)
        self.buttons["red"].place(x=150, y=287)
        self.buttons["yellow"].place(x=370, y=287)


    def start_game(self):
        """

        :return:
        """
        self.user_choices_array = []
        self.colors_array = []
        self.options_array = ["green", "blue", "red", "yellow"]
        self.thread = Thread(target=self.game_manage)
        self.thread.start()


    def add_color(self, color):
        """

        :param color:
        :return:
        """
        self.user_choices_array.append(color)


    def check_colors(self):
        """

        :return:
        """

        # looping through the colors array
        for index, color in enumerate(self.colors_array):

            # checking if the colors does not match
            if self.user_choices_array[index - 1] != self.colors_array[index - 1]:
                return False

        return True


    def game_manage(self):
        """

        :return:
        """

        # while the user is correct
        while (self.check_colors()):

            # looping through the colors to present them
            for choice in self.user_choices_array:
                self.buttons[choice].config(bg="white")
                time.sleep(1)
                self.buttons[choice].config(bg=choice)

            new_color = random.choice(self.options_array)

            self.buttons[new_color].config(bg="white")
            time.sleep(1)
            self.buttons[new_color].config(bg=new_color)

            self.colors_array.append(new_color)

            self.user_choices_array = []

            # waiting for the users to click all the options
            while len (self.user_choices_array) != len (self.colors_array):
                pass

        messagebox.showerror("Game ended",  f"\nYou have lost, you succeded remembering {len(self.colors_array) - 1} colors")
        time.sleep(1)


def main():
    app = MainApplication()
    app.geometry("700x550")
    app.mainloop()


if __name__ == "__main__":
    main()
