from collections import OrderedDict
import os
import re

import chaipy.common as common

def parse_hresults(hresults_fname):
    """ Parse content of HResults output. Return a dict of word-level results.
    """
    regex = '.*Corr=([\.0-9]+).*Acc=([-\.0-9]+).*H=([0-9]+).*D=([0-9]+).*S=([0-9]+).*I=([0-9]+).*'
    results = OrderedDict()
    with open(hresults_fname, 'r') as f:
        for line in f.readlines():
            if line.startswith('WORD'):
                line = line.strip()
                match = re.match(regex, line)
                if match:
                    results['Corr'] = float(match.group(1))
                    results['Acc'] = float(match.group(2))
                    results['H'] = int(match.group(3))
                    results['D'] = int(match.group(4))
                    results['S'] = int(match.group(5))
                    results['I'] = int(match.group(6))
                    break
                else:
                    raise ValueError('Failed to match results: %s' % line)
    return results

def read_mlf(mlf_fname, rm_ext=True, rm_path=True, allow_empty=False):
    """ Read content of MLF into a mapping from utterance name to content lines
    """
    mlf = OrderedDict()
    utt_name = None
    lines = []
    with open(mlf_fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif line.startswith("\""):
                utt_name = line.replace("\"", "")
                utt_name = common.rm_ext(utt_name) if rm_ext else utt_name
                utt_name = os.path.basename(utt_name) if rm_path else utt_name
            elif line == ".":
                if utt_name is not None:
                    if allow_empty or len(lines) > 0:
                        mlf[utt_name] = lines
                lines = []
            else:
                lines.append(line)
    return mlf

def read_segmented_mfccs(mfc_fname, rm_ext=True, rm_path=True, step_ms=10):
    """ Read a segmented MFCC scp file in HTK format, creating a 
    mapping from utterance names to SegmentedMFCC objects.
    """
    segmented_mfccs = OrderedDict()
    with open(mfc_fname, 'r') as f:
        for line in f:
            line = line.strip()
            ary = line.split('=')
            common.CHK_LEN(ary, 2)
            # Utterance name
            utt_name = common.rm_ext(ary[0]) if rm_ext else ary[0]
            utt_name = os.path.basename(utt_name) if rm_path else utt_name
            # Segmented MFCC
            open_bracket_idx = ary[1].rfind('[')
            common.CHK_NEQ(open_bracket_idx, -1)
            comma_idx = ary[1].rfind(',')
            common.CHK_NEQ(comma_idx, -1)
            close_bracket_idx = ary[1].rfind(']')
            common.CHK_NEQ(close_bracket_idx, -1)
            mfc = ary[1][:open_bracket_idx]
            start_frame = int(ary[1][open_bracket_idx + 1:comma_idx])
            end_frame = int(ary[1][comma_idx + 1:close_bracket_idx])
            segmented_mfccs[utt_name] = \
                    SegmentedMFCC(mfc, start_frame, end_frame, step_ms=step_ms)
    return segmented_mfccs

class SegmentedMFCC(object):
    """
    Contains path to original MFCC and start and end frame numbers.
    """
    def __init__(self, mfc, start_frame, end_frame, step_ms=10):
        """
        :type mfc: str
        :param mfc: Path to original MFCC

        :type start_frame: int
        :param start_frame: Start frame

        :type end_frame: int
        :param end_frame: End frame

        :type step_ms: int
        :param step_ms: Step size in ms
        """
        self.mfc = mfc
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step_ms = step_ms

    def get_start_time(self, unit='s'):
        """ Return start time in seconds ('s') or milliseconds ('ms').
        """
        common.CHK_VALS(unit, ['s', 'ms'])
        start_time_ms = float(self.start_frame * self.step_ms)
        return start_time_ms if unit == 'ms' else common.ms_to_s(start_time_ms)

    def get_end_time(self, unit='s'):
        """ Return end time in seconds ('s') or milliseconds ('ms').
        """
        common.CHK_VALS(unit, ['s', 'ms'])
        end_time_ms = float((self.end_frame + 1) * self.step_ms)
        return end_time_ms if unit == 'ms' else common.ms_to_s(end_time_ms)

    def get_times(self, unit='s'):
        """ Return a tuple containing start and end times in 's' or 'ms'.
        """
        return (self.get_start_time(unit=unit), self.get_end_time(unit=unit))
