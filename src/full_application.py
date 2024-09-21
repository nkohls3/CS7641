# This file contains a top-to-bottom script that demonstrates our project.

# ===== IMPORTS ===== #
from latex_writer import LaTeXWriter, Character
# TODO: import classifier models

# ===== INITIALIZE ===== #
writer = LaTeXWriter()
# TODO: other init

# ===== CLASSIFY ===== #
# this section transcribes a single equation.
# to loop, just wrap this whole block in a for loop

# helper functions
def load(equation):
    # idk if this is necessary
    # loads equation image
    
    loaded_equation = None
    return loaded_equation

def separate(equation_image):
    # separates equation image into individual characters
    # keeps track of the position of each character within the
    # original equation image, for reassembly in LaTeX

    separated_character_images = [None] # N x 1 array of images
    separated_character_positions = [[0, 0]] # N x 2 array of [x, y] positions
    return separated_character_images, separated_character_positions

def classify(character_image):
    # run whatever classifier

    classification = "a"
    return classification

# run classification
characters_in_equation = []
equation = "some/path"
equation_img = load(equation)
separated_character_images, separated_character_positions = separate(equation_img)

for i in range(len(separated_character_images)):
    character_image = separated_character_images[i]
    character_pos = separated_character_positions[i]
    character_id = classify(character_image)
    character_object = Character(character_id, character_pos)
    characters_in_equation.append(character_object)

# ===== TRANSCRIBE ===== #
equation_latex = writer.write_equation(characters_in_equation)
print(equation_latex)
