from csv import reader
from latex_writer import LaTeXWriter, Character
writer = LaTeXWriter()

# test characters
height = 0
H = Character("H", [0, height])
open_1 = Character("(", [1, height])
x_1 = Character("x", [2, height])
close_1 = Character(")", [3, height])
equals = Character("=", [4, height])
minus = Character("-", [5, height])
sigma = Character("sum", [6, height])
p_1 = Character("p", [7, height])
open_2 = Character("(", [8, height])
x_2 = Character("x", [9, height])
close_2 = Character(")", [10, height])
l = Character("l", [11, height])
o = Character("o", [12, height])
g = Character("g", [13, height])
p_2 = Character("p", [14, height])
open_3 = Character("(", [15, height])
x_3 = Character("x", [16, height])
close_3 = Character(")", [17, height])

eq4 = [H, open_1, x_1, close_1, equals, minus, sigma, p_1, open_2, x_2, close_2, l, o, g, p_2, open_3, x_3, close_3]

csv_dir = "/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/test_equations/"
for j in range(5):
    # get equation csv number
    number = str(int(j + 1))
    file_num = "Eq4_" + number
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

    for i in range(len(eq4)):
        char = eq4[i]
        pos = character_positions[i]
        char.pos = pos
        eq4[i] = char

    eq5_string = writer.write_equation(eq4)
    print(eq5_string)
