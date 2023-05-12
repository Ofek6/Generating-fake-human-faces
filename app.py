from Algorithms import Algorithms
from Gui import GUI

class App:
    def __init__(self):
        self.algorithms = Algorithms()
        self.gui = GUI(self)

    def show_choice(self):
        choice = self.gui.v.get()
        if choice == 101:
            self.gui.image_label.config(image=self.algorithms.personal_details)
        elif choice == 102:
            self.gui.image_label.config(image=self.algorithms.architecture_of_network)
        elif choice == 103:
            self.gui.image_label.config(image=self.algorithms.loss_scale)
        elif choice == 104:
            image = self.algorithms.generate_and_display_image(self.algorithms.generator0)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image
        elif choice == 105:
            image = self.algorithms.generate_and_display_image(self.algorithms.generator20)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image
        elif choice == 106:
            image = self.algorithms.generate_and_display_image(self.algorithms.generator40)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image
        elif choice == 107:
            image = self.algorithms.generate_and_display_image(self.algorithms.generator60)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image
        elif choice == 108:
            image = self.algorithms.generate_and_display_image(self.algorithms.generator80)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image
        elif choice == 109:
            image = self.algorithms.generate_and_display_image(self.algorithms.generatorAll)
            self.gui.image_label.config(image=image)
            self.gui.image_label.image = image


if __name__ == "__main__":
    app = App()
    app.algorithms.load_images()
    app.gui.window.mainloop()
