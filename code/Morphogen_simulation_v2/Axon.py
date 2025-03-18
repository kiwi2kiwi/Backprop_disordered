import Cell_v2

class Axon():
    def __init__(self, parent, child, excite_inhibit, cell_space):
        self.name = cell_space.Axon_counter
        cell_space.Axon_counter += 1
        cell_space.Axons[self.name] = self
        self.parent = parent
        self.parent.children[child.name] = child
        self.parent.Axons[self.name]=self
        self.child = child
        self.child.parents[parent.name] = parent
        # self.child.Axons[self.name] = self # children dont get axons to parents
        self.excite_inhibit = excite_inhibit # excitatory or inhibitory cell

