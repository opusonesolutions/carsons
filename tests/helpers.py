class LineModel:
    def __init__(self, conductors):
        self._resistance = {}
        self._geometric_mean_radius = {}
        self._wire_positions = {}
        self._phases = {}

        for phase, (r, gmr, (x, y)) in conductors.items():
            self._resistance[phase] = r
            self._geometric_mean_radius[phase] = gmr
            self._wire_positions[phase] = (x, y)
            self._phases = sorted(list(conductors.keys()))

    @property
    def resistance(self):
        return self._resistance

    @property
    def geometric_mean_radius(self):
        return self._geometric_mean_radius

    @property
    def wire_positions(self):
        return self._wire_positions

    @property
    def phases(self):
        return self._phases
