from csv import reader
from latex_writer import LaTeXWriter, Character
writer = LaTeXWriter()

# test characters
height = 0
y = Character("y", [1, height])
equals = Character("=", [2, height])
m = Character("m", [3, height])
x = Character("x", [4, height])
plus = Character("+", [5, height])
b = Character("b", [6, height])

eq2 = [y, equals, m, x, plus, b]

csv_dir = "/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/test_equations/"
for j in range(5):
    # get equation csv number
    number = str(int(j + 1))
    file_num = "Eq2_" + number
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

    for i in range(len(eq2)):
        char = eq2[i]
        pos = character_positions[i]
        char.pos = pos
        eq2[i] = char

    eq5_string = writer.write_equation(eq2)
    print(eq5_string)
