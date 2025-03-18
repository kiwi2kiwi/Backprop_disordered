import Cell_v2

class Axon():
    def __init__(self, parent, child, excite_inhibit, cell_space):
        self.name = cell_space.Axon_counter
        cell_space.Axon_counter += 1
        self.parent = parent
        self.parent.Axon
        self.child = child
        self.excite_inhibit = excite_inhibit # excitatory or inhibitory cell
        self.parent.cell_space.addAxon(self)

