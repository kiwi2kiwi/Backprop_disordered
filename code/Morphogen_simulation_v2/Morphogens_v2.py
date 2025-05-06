class Morphogens_v2():
    def __init__(self, amount, cell_space, cell_unique=False):
        self.name = cell_space.Morphogen_counter # len(cell_space.Morphogens.keys())
        cell_space.Morphogen_counter += 1
        cell_space.Morphogens[self.name] = self
        self.cell_space = cell_space
        self.amount = amount
        self.cells = {}
        self.cell_unique = cell_unique # if the morphogen is the unique cell address that cannot be deleted

    # TODO create a delete cell function
    def add_cell(self, cell_name):
        # if cell_name == 9:
        #     print("stop")
        self.cells[cell_name] = self.cell_space.Cells[cell_name]

    # delete morphogen from the cell_space and also clear its associated cells
    def delete_morpho(self):
        if not self.cell_unique:
            self.cell_space.Morphogens.pop(self.name)
            for c in self.cells:
                c.del_morphogen(self)

            # self.cells.pop(self.name)

