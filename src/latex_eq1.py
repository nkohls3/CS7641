from csv import reader
from latex_writer import LaTeXWriter, Character
writer = LaTeXWriter()

# test characters
height = 0
one_1 = Character("1", [1, height])
plus = Character("+", [2, height])
one_2 = Character("1", [3, height])
equals = Character("=", [4, height])
two = Character("2", [5, height])

eq1 = [one_1, plus, one_2, equals, two]

csv_dir = "/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/test_equations/"
for j in range(5):
    # get equation csv number
    number = str(int(j + 1))
    file_num = "Eq1_" + number
    file_name = file_num + "/" + file_num + ".csv"
    csv_path = csv_dir + file_name

    # get character locations
    character_positions = []
    with open(csv_path, newline="") as f:
        r = reader(f, delimiter=",")
        for row in r:
            x = row[0].replace("(", "")
            y = row[1].replace(")", "")
            character_positions.append((int(x), int(y)))

    for i in range(len(eq1)):
        char = eq1[i]
        pos = character_positions[i]
        char.pos = pos
        eq1[i] = char

    eq5_string = writer.write_equation(eq1)
    print(eq5_string)
