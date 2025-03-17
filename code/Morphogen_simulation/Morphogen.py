class Morphogen():
    def __init__(self, name, rule):
        # rule is a function whose effect that is multiplied by the morphogen strength
        self.name = name
        self.rule = rule



    def deteriorate(self, morphogen_abundancy, cell):
        # print("deteriorating cell ", cell.name)
        morphogen_abundancy = morphogen_abundancy - 0.1
        if morphogen_abundancy < 0:
            morphogen_abundancy = 0
        return morphogen_abundancy

