# Notes:
# ```
# need to lower() for all capital/lowercase confusion pairs, e.g. C and X
# blobs on top of each other - that indicates "=" or 
# 
# functions for me: log, exp, sqrt, trig
# check if the sequence 'l o g' is in the string
# 
# expressions we plan to use:
# 1 + 1 = 2
# y = m x + b
# A^2 + B^2 = C^2
# cos^2(x) + sin^2(x) = 1
# H(x) = - \sum p(x) log(p(x)) [ entropy equation ]
# ```

import numpy as np

class Character:
    def __init__(self, character_id, pos):
        self.character_id = character_id
        self.pos = pos

class LaTeXFeatures:
    def __init__(self):
        self._frac = '\ frac'.replace(" ","")
    
    def curly(self, x):
        return "{" + str(x.character_id) + "}"

    def frac(self, x, y):
        return '{}{}{}'.format(self._frac, self.curly(x), self.curly(y))

    def exp(self, x):
        return "^" + self.curly(x)

class LaTeXWriter:
    def __init__(self):
        self.features = LaTeXFeatures()
        self.control_sequences = ["sin", "cos", "sum", "log"]
        self.exp_excluded_chars = ["+", "-", "=", "."] # exclude small characters - may be buggy for "-"
        self.exp_height = 50

    def sort_characters_lr(self, characters):
        character_positions = np.array([character.pos for character in characters])
        sorted_ids = np.argsort(character_positions[:, 0])
        new_characters = []
        for i in sorted_ids:
            new_characters.append(characters[i])
            
        return new_characters

    def get_character_sequence(self, characters):
        character_sequence = ""
        for character in characters:
            character_sequence += character.character_id
        
        return character_sequence

    def get_character_start_indices_in_sequence(self, characters):
        # useful when characters have character_ids longer than 1
        start_indices = []
        i = -1
        for character in characters:
            j = len(character.character_id)
            i += j
            start_indices.append(i)
        return start_indices

    def check_for_sequence(self, characters, sequence):
        character_sequence = self.get_character_sequence(characters)
        sequence_start = -1
        single_char = False

        if sequence in character_sequence:
            pos = character_sequence.find(sequence)
            # print("Sequence " + sequence + " found at position " + str(pos))
            sequence_start = pos
        
            # check if sequence is in any single character
            for i in range(len(characters)):
                character = characters[i]
                if character.character_id == sequence:
                    sequence_start = i - 1
                    single_char = True
        
        return sequence_start, single_char
    
    def write_equation(self, characters):
        characters = self.sort_characters_lr(characters)
        character_start_indices = self.get_character_start_indices_in_sequence(characters)
        eqn_string = "$"

        # check for control sequences
        for control in self.control_sequences:
            sequence_start, single_char = self.check_for_sequence(characters, control)

            # check if sequence found; -1 indicates not found
            # extra logic for handling if a control sequence is inside a single char
            if sequence_start >= 0:
                char_indices = np.where(np.array(character_start_indices) == sequence_start)[0]
                for i in range(len(char_indices)):
                    char_index = np.asscalar(char_indices[i])
                    if single_char:
                        char_index += 1
                    new_id = "\\" + control
                    new_char = Character(new_id, characters[char_index].pos)
                    n_characters = len(control)
                    end_start = char_index + n_characters - len(control) + 1 if single_char else char_index + n_characters
                    characters = characters[:char_index] + [new_char] + characters[end_start:]

            # recompute starts
            character_start_indices = self.get_character_start_indices_in_sequence(characters)

        for i in range(len(characters)):
            character = characters[i]

            (x, y) = character.pos

            if i > 0:
                prev_character = characters[i - 1]
                (x_prev, y_prev) = prev_character.pos

                # check if exponent
                if -(y - y_prev) > self.exp_height and characters[i].character_id not in self.exp_excluded_chars:
                    eqn_string += self.features.exp(character)
                else:
                    eqn_string += character.character_id

            else:
                eqn_string += character.character_id
            
            eqn_string += " "

        eqn_string += "$"
        return eqn_string
