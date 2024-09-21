from csv import reader
from latex_writer import LaTeXWriter, Character
writer = LaTeXWriter()

# test characters
height = 0
c = Character("c", [0, height])
o = Character("o", [1, height])
s_1 = Character("s", [2, height])
two_1 = Character("2", [3, height])
open_1 = Character("(", [4, height])
x_1 = Character("x", [5, height])
close_1 = Character(")", [6, height])
plus = Character("+", [7, height])
s_2 = Character("s", [8, height])
i = Character("i", [9, height])
n = Character("n", [10, height])
two_2 = Character("2", [11, height])
open_2 = Character("(", [12, height])
x_2 = Character("x", [13, height])
close_2 = Character(")", [14, height])
equals = Character("=", [15, height])
one = Character("1", [16, height])

eq3 = [c, o, s_1, two_1, open_1, x_1, close_1, plus, s_2, i, n, two_2, open_2, x_2, close_2, equals, one]

csv_dir = "/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/test_equations/"
for j in range(5):
    # get equation csv number
    number = str(int(j + 1))
    file_num = "Eq3_" + number
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

    for i in range(len(eq3)):
        char = eq3[i]
        pos = character_positions[i]
        char.pos = pos
        eq3[i] = char

    eq5_string = writer.write_equation(eq3)
    print(eq5_string)
