def make_two_children_below(cell, amount):
    if amount >= 0.2:
        cell.mitosis_counter = 2
        if cell.name == 0:
            print("2 mitosis for cell 0")

    if amount < 0.2 and amount > 0:
        cell.morphogens["leg"][1] = 0.5



def elongate_down(cell, amount):
    if amount >= 0.2:
        cell.mitosis_counter = 1



