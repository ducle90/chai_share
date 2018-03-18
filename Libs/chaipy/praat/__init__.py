from collections import OrderedDict

import chaipy.common as common

class Interval(object):
    """
    Represent a Praat interval.
    """
    def __init__(self, xmin=0.0, xmax=0.0):
        """
        :type xmin: float
        :param xmin: Start time (in s)

        :type xmax: float
        :param xmax: End time (in s)
        """
        self.xmin = xmin
        self.xmax = xmax

    def get_start_time(self, unit='s'):
        """ Return start time in seconds ('s') or milliseconds ('ms').
        """
        common.CHK_VALS(unit, ['s', 'ms'])
        return self.xmin if unit == 's' else common.s_to_ms(self.xmin)

    def get_end_time(self, unit='s'):
        """ Return end time in seconds ('s') or milliseconds ('ms').
        """
        common.CHK_VALS(unit, ['s', 'ms'])
        return self.xmax if unit == 's' else common.s_to_ms(self.xmax)

    def get_times(self, unit='s'):
        """ Return a tuple containing start and end times in 's' or 'ms'.
        """
        return (self.get_start_time(unit=unit), self.get_end_time(unit=unit))

class TextGrid(Interval):
    """
    Represent a Praat TextGrid file.
    """
    # Data reading mode
    READ_TEXTGRID = 0
    READ_ITEM = 1
    READ_INTERVAL = 2

    def __init__(self, xmin=0.0, xmax=0.0):
        """
        :type xmin: float
        :param xmin: Start time (in s)

        :type xmax: float
        :param xmax: End time (in s)
        """
        super(TextGrid, self).__init__(xmin=xmin, xmax=xmax)
        # Mapping from name to item (a.k.a tier)
        self.items = OrderedDict()

    @classmethod
    def from_file(cls, textgrid_fname):
        """ Initialize from TextGrid file.
        """
        textgrid = cls()
        # Buffer variables
        read_mode = cls.READ_TEXTGRID
        item = None
        interval = None
        # Begin reading line by line
        with open(textgrid_fname, 'r') as f:
            for line in f:
                line = line.strip()
                # Read general TextGrid info
                if read_mode == cls.READ_TEXTGRID:
                    if line.startswith('xmin = '):
                        textgrid.xmin = float(cls.get_content(line))
                    elif line.startswith('xmax = '):
                        textgrid.xmax = float(cls.get_content(line))
                    elif line.startswith('item ['):
                        read_mode = cls.READ_ITEM
                # Read general tier info
                elif read_mode == cls.READ_ITEM:
                    if line.startswith('class = '):
                        cname = cls.get_content(line)
                        if cname == 'IntervalTier':
                            item = IntervalTier()
                        else:
                            raise ValueError('Invalid tier class: {}'.format(cname))
                    elif line.startswith('name = '):
                        item.name = cls.get_content(line)
                    elif line.startswith('xmin = '):
                        item.xmin = float(cls.get_content(line))
                    elif line.startswith('xmax = '):
                        item.xmax = float(cls.get_content(line))
                    elif line.startswith('intervals'):
                        read_mode = cls.READ_INTERVAL
                # Read interval
                elif read_mode == cls.READ_INTERVAL:
                    if line.startswith('intervals ['):
                        if interval is not None:
                            item.intervals.append(interval)
                        interval = item.new_interval()
                    elif line.startswith('xmin = '):
                        interval.xmin = float(cls.get_content(line))
                    elif line.startswith('xmax = '):
                        interval.xmax = float(cls.get_content(line))
                    elif line.startswith('text = '):
                        interval.text = cls.get_content(line)
                    elif line.startswith('item ['):
                        if interval is not None:
                            item.intervals.append(interval)
                        textgrid.items[item.name] = item
                        interval = None
                        read_mode = cls.READ_ITEM
                else:
                    raise ValueError('Invalid read mode: {}'.format(read_mode))
        # Final flush
        if item is not None:
            if interval is not None:
                item.intervals.append(interval)
            textgrid.items[item.name] = item
        return textgrid

    @classmethod
    def get_content(cls, line):
        return line[line.find('=')+1:].strip().replace('\"', '')

class IntervalTier(Interval):
    """
    Represent a Praat IntervalTier
    """
    def __init__(self, xmin=0.0, xmax=0.0, name=""):
        """
        :type xmin: float
        :param xmin: Start time (in s)

        :type xmax: float
        :param xmax: End time (in s)

        :type name: str
        :param name: Name of this tier
        """
        super(IntervalTier, self).__init__(xmin=xmin, xmax=xmax)
        self.name = name
        # List of intervals in this tier
        self.intervals = []

    @classmethod
    def new_interval(cls):
        """ Return a new empty interval object for this tier
        """
        return TextInterval()

class TextInterval(Interval):
    """
    An interval with text label
    """
    def __init__(self, xmin=0.0, xmax=0.0, text=""):
        """
        :type xmin: float
        :param xmin: Start time (in s)

        :type xmax: float
        :param xmax: End time (in s)

        :type text: str
        :param text: Label of this interval
        """
        super(TextInterval, self).__init__(xmin=xmin, xmax=xmax)
        self.text = text

    def __str__(self):
        return '<{}> (xmin={} xmax={} text=\"{}\")'.format(
                TextInterval.__name__, self.xmin, self.xmax, self.text
        )

    def __repr__(self):
        return self.__str__()
