import numpy as np
import PIL.Image
import io
from PIL import Image, ImageTk
import tensorflow as tf
from tkinter import PhotoImage  # Add this line

class Algorithms:
    def __init__(self):
        self.load_generators()

    def load_generators(self):
        self.generator = tf.keras.models.load_model("generator (1).h5")
        self.generator0 = tf.keras.models.load_model("newgenerator0.h5")
        self.generator20 = tf.keras.models.load_model("newgenerator20.h5")
        self.generator40 = tf.keras.models.load_model("newgenerator40.h5")
        self.generator60 = tf.keras.models.load_model("newgenerator60.h5")
        self.generator80 = tf.keras.models.load_model("newgenerator80.h5")
        self.generatorAll = tf.keras.models.load_model("newgeneratorAll.h5")

    def generate_and_display_image(self, generator_name):
        inputs = np.random.rand(1, 100)
        generated_image = generator_name.predict(inputs)
        generated_image = (generated_image + 1) * 127.5
        generated_image = generated_image.astype(np.uint8)
        pil_image = PIL.Image.fromarray(generated_image[0, :, :, :], mode='RGB')
        pil_image = pil_image.resize((512, 512), PIL.Image.ANTIALIAS)
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='png')
        image_bytes.seek(0)
        image = PhotoImage(data=image_bytes.read())
        return image

    def load_images(self):
        loss_scale = Image.open("‏‏new_loss_31.12.22 - small.png")
        architecture_of_network = Image.open("‏‏network_design_31.12.22 - small.png")
        personal_details = Image.open("personal_details_new_for_gui.png")

        resized_loss_scale = loss_scale.resize((512, 512))
        resized_architecture_of_network = architecture_of_network.resize((512, 512))
        resized_personal_details = personal_details.resize((512, 512))

        self.loss_scale = ImageTk.PhotoImage(resized_loss_scale)
        self.architecture_of_network = ImageTk.PhotoImage(resized_architecture_of_network)
        self.personal_details = ImageTk.PhotoImage(resized_personal_details)


