import Cell_v2

class Axon():
    def __init__(self, parent, child, excite_inhibit):
        self.parent = parent
        self.child = child
        self.excite_inhibit = excite_inhibit # excitatory or inhibitory cell
        self.parent.cell_space.addAxon(self)

