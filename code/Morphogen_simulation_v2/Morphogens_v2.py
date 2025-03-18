class Morphogens_v2():
    def __init__(self, amount, cell_space, cell_unique=False):
        self.name = cell_space.Morphogen_counter
        cell_space.Morphogen_counter += 1
        cell_space.Morphogens[self.name] = self
        self.cell_space = cell_space
        self.amount = amount
        self.cells = []
        self.cell_unique = cell_unique # if the morphogen is the unique cell address that cannot be deleted


    def delete_morpho(self):
        if not self.cell_unique:
            self.cell_space.Morphogens.pop(self.name)


    # def deteriorate(self, morphogen_abundancy, cell):
    #     # print("deteriorating cell ", cell.name)
    #     morphogen_abundancy = morphogen_abundancy - 0.1
    #     if morphogen_abundancy < 0:
    #         morphogen_abundancy = 0
    #     return morphogen_abundancy

