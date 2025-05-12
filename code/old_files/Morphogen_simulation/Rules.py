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


class Rule():
    def __init__(self, rule):
        self.rule = rule
        self.address

        #TODO
        # can connect to multiple addresses
        # needs those addresses
        # needs to know
        # can create new cell

    def rule(self, morphogen, threshold, executing_cell):
        if morphogen >= threshold:
            executing_cell


    def create_new_cell(self):
        # do i copy the rules?

        # when divided, put the child cell address here and create a rule that connects to the address marker of the child cell

        # TODO connects to the created cell
        pass

    def connect(self):
        # TODO connects to the created cell
        pass