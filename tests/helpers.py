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


class ConcentricLineModel:
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
                self._neutral_strand_resistance[phase] = val['neutral_strand_resistance']   # noqa 401
                self._neutral_strand_diameter[phase] = val['neutral_strand_diameter']       # noqa 401
                self._diameter_over_neutral[phase] = val['diameter_over_neutral']           # noqa 401
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


class MultiLineModel:
    def __init__(self, conductors, is_secondary=False):
        self._resistance = {}
        self._geometric_mean_radius = {}
        self._wire_positions = {}
        self._radius = {}
        self._insulation_thickness = {}

        for phase, val in conductors.items():
            self._resistance[phase] = val['resistance']
            self._geometric_mean_radius[phase] = val['gmr']
            self._wire_positions[phase] = val['wire_positions']
            self._radius[phase] = val['radius']
            self._insulation_thickness[phase] = val['insulation_thickness']

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
    def radius(self):
        return self._radius

    @property
    def insulation_thickness(self):
        return self._insulation_thickness
