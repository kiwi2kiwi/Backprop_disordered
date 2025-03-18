class Morphogens_v2():
    def __init__(self, name, amount):
        self.name = name
        self.amount = amount
        self.cells = []



    # def deteriorate(self, morphogen_abundancy, cell):
    #     # print("deteriorating cell ", cell.name)
    #     morphogen_abundancy = morphogen_abundancy - 0.1
    #     if morphogen_abundancy < 0:
    #         morphogen_abundancy = 0
    #     return morphogen_abundancy

