class _Phase:
    """ Class for representing the phase of a wire."""
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'Phase({self.label})'

    def __eq__(self, other):
        return other == self.label

    def __hash__(self):
        return hash(self.label)

    def __lt__(self, other):
        return self.label < other.label


class Neutral(_Phase):
    """ Class specifically for itemizing neutral conductors """
    def __init__(self, label):
        if not label.startswith("N"):
            raise ValueError(
                f"Neutral label must start with 'N'; got '{label}'"
            )
        super().__init__(label)


A = _Phase("A")
B = _Phase("B")
C = _Phase("C")
N = Neutral("N")
