class LineModel:
    def __init__(self, conductors):
        self._resistance = {}
        self._geometric_mean_radius = {}
        self._wire_positions = {}
        self._phases = {}
        self._neutral_strand_gmr = {}
        self._neutral_strand_resistance = {}
        self._neutral_strand_diameter = {}
        self._diameter_over_neutral = {}
        self._neutral_strand_count = {}

        for phase, val in conductors.items():
            if 'N' in phase:
                self._neutral_strand_gmr[phase] = val['neutral_strand_gmr']
                self._neutral_strand_resistance[phase] = val['neutral_strand_resistance']
                self._neutral_strand_diameter[phase] = val['neutral_strand_diameter']
                self._diameter_over_neutral[phase] = val['diameter_over_neutral']
                self._neutral_strand_count[phase] = val['neutral_strand_count']
            else:
                self._resistance[phase] = val['resistance']
                self._geometric_mean_radius[phase] = val['gmr']
                self._wire_positions[phase] = val['wire_positions']
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

    @property
    def neutral_strand_gmr(self):
        return self._neutral_strand_gmr

    @property
    def neutral_strand_resistance(self):
        return self._neutral_strand_resistance

    @property
    def neutral_strand_diameter(self):
        return self._neutral_strand_diameter

    @property
    def diameter_over_neutral(self):
        return self._diameter_over_neutral

    @property
    def neutral_strand_count(self):
        return self._neutral_strand_count
