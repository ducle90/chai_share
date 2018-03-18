# Copyright 2015    Duc Le  University of Michigan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import os
from chaipy import common, io
from chaipy.data import fill_vals, fill_ary, accum_val, get_num_items, partition
from bisect import bisect_right
from collections import OrderedDict


class TemporalData(object):
    """
    This class is an abstraction for a collection of temporal data, such as
    audio or video. The data can be broken down into 'utterances', where
    each utterance is represented by a multi-dimensional time series. The
    utterances can be optionally grouped in certain ways, e.g. by speakers.
    We currently assume that the entire dataset can be stored in memory. We
    will relax this assumption in future versions should the need arise.

    Supported features:
        * Index dataset by utterance name and utterance index
        * Index dataset by utterance names and utterance indices
        * Index dataset by frame index and frame indices
        * Inclusion of optional utterance features (e.g. i-vectors)
        * Efficient leave-one-subject-out partitioning
        * Efficient partitioning by utterances
        * On the fly context padding (for DNN training)
        * Efficient method to obtain frame-level data (for DNN training)
        * Initialization from data directory stored on disk
        * Initialization from Kaldi features stored on disk (ark+scp)
    """
    def __init__(self, data, utt_names, end_indices, start_indices=None,
                 dummy_index=None, utt_labels=None, num_utt_classes=None,
                 frame_labels=None, labels_dtype=numpy.int32,
                 num_frame_classes=None, utt_feats_dict=None,
                 utt_feats_ary=None, utt2group=None, context=0, padding='zero',
                 seed=None, dtype=numpy.float32, context_map=None,
                 readonly=False, dynamic_context=True):
        """
        :type data: numpy.array
        :param data: 2D data array

        :type utt_names: list
        :param utt_names: List of utterance names

        :type end_indices: list
        :param end_indices: List containing the ending index in the data array
            for each utterance in utt_names. Specifically, the last index of
            utt_names[i] in data is end_indices[i] - 1.

        :type start_indices: list
        :param start_indices: (optional) List containing the starting index in
            the data array for each utterance in utt_names. Specifically, the
            first index of utt_names[i] in data is start_indices[i]. If not
            specified, start_indices will be automatically derived from
            end_indices, assuming utterances are next to each other in data.

        :type dummy_index: int
        :param dummy_index: (optional) Index of dummy vector in data. If not
            specified, a zero vector will be added automatically to data.

        :type utt_labels: list
        :param utt_labels: (optional) List of utterance labels containing
            floats or integers that fall in range(0, num_utt_classes).

        :type num_utt_classes: int
        :param num_utt_classes: (optional) Number of utterance-level classes.
            If not specified, will be determined automatically from utt_labels.

        :type frame_labels: list
        :param frame_labels: (optional) List of frame labels containing
            floats or integers that fall in range(0, num_frame_classes). If
            utt_labels is set but frame_labels is not, frame labels are
            assumed to have the same value as utterance labels.

        :type labels_dtype: numpy.dtype
        :param labels_dtype: (optional) Labels type.

        :type num_frame_classes: int
        :param num_frame_classes: (optional) Number of frame-level classes. If
            not specified, will be determined automatically from frame_labels.

        :type utt_feats_dict: dict
        :param utt_feats_dict: (optional) Mapping from utterance name to
            utterance-level features. If present, frame-level features will
            be augmented with utterance-level vector.

        :type utt_feats_ary: numpy.array
        :param utt_feats_ary: (optional) Expanded form of utterance-level
            features. If utt_feats_dict is specified but utt_feats_ary is not,
            this is automatically generated. Don't set this value unless you
            know what you're doing! This is intended for dataset trimming.

        :type utt2group: dict
        :param utt2group: (optional) Dictionary that maps utterance to group

        :type context: int
        :param context: (optional) The number of context frames to add to each
            side of the current frame. For example, context=5 means that the
            current frame will be augmented with 5 frames to the left and 5
            frames to the right. Internally, data is not expanded. It is only
            expanded on-the-fly at function calls.

        :type padding: str
        :param padding: (optional) The type of padding to use for out-of-bound
            frames. This only applies when context > 0. Available options:
                'zero' - Pad with zeros
                'replicate' - Replicate nearest frame (common in ASR)

        :type seed: int
        :param seed: (optional) Random seed that controls the shuffling of data

        :type dtype: numpy.dtype
        :param dtype: (optional) Data type. Use numpy.float32 for GPU training.

        :type context_map: numpy.array
        :param context_map: (optional) Context map. Don't set this value unless
            know what you're doing! This is intended for dataset trimming.

        :type readonly: bool
        :param readonly: (optional) If True, denote that this dataset is
            read-only. This is useful when the dataset is sharing data with
            other object(s), e.g. trimmed datasets. The flag will prevent
            potentialy disruptive functions from running by raising an error.

        :type dynamic_context: bool
        :param dynamic_context: (optional) If True, don't pre-fill context map.
            Instead, context indices will be generated on the fly.
        """
        self.readonly = False   # Allow initial setters to run
        # Set data array
        self.__set_data(data, dummy_index, dtype)
        # Set utterance metainfo
        self.__set_utt_metainfo(utt_names, start_indices, end_indices)
        # Set optional arguments
        self.set_labels(utt_labels, num_utt_classes,
                        frame_labels, num_frame_classes, labels_dtype)
        self.set_utt_feats(utt_feats_dict, utt_feats_ary)
        self.set_utt2group(utt2group)
        self.set_context(context, padding=padding, context_map=context_map,
                         dynamic_context=dynamic_context)
        self.set_seed(seed)
        self.readonly = readonly

    def __set_data(self, data, dummy_index, dtype):
        common.CHK_DIM(data, 2)
        if dummy_index is None:
            # We keep a dummy vector of all zeros at the last row. This allows
            # efficient inerstion of zero vectors when making context windows.
            self.data = numpy.zeros((data.shape[0] + 1, data.shape[1]),
                                    dtype=dtype)
            self.data[0:data.shape[0]] = data
            self.dummy_index = data.shape[0]
        else:
            self.data = numpy.asarray(data, dtype=dtype)
            self.dummy_index = dummy_index

    def __set_utt_metainfo(self, utt_names, start_indices, end_indices):
        common.CHK_EQ(len(utt_names), len(end_indices))
        self.utt_names = numpy.asarray(utt_names, dtype=numpy.str)
        if start_indices is None:
            self.start_indices = numpy.zeros(len(end_indices), dtype=numpy.int)
            self.start_indices[1:] = end_indices[0:-1]
        else:
            common.CHK_EQ(len(start_indices), len(end_indices))
            self.start_indices = numpy.asarray(start_indices, dtype=numpy.int)
        self.end_indices = numpy.asarray(end_indices, dtype=numpy.int)
        # Map from utterance name to utterance index
        self.utt_to_index = common.make_position_index(utt_names)
        # Array of utterance indices
        self.__utt_indices = numpy.arange(len(utt_names))
        # Array of frame indices
        frame_indices = []
        for start_index, end_index in zip(self.start_indices, self.end_indices):
            frame_indices.extend(numpy.arange(start_index, end_index))
        self.__frame_indices = numpy.asarray(frame_indices, dtype=numpy.int)

    def set_labels(self, utt_labels=None, num_utt_classes=None,
                   frame_labels=None, num_frame_classes=None,
                   labels_dtype=numpy.int32):
        """ Because frame labels may depend on utterance labels, we have to
        set them jointly. Warning, labels will be overwritten each time!
        """
        assert not self.readonly
        # First, process utterance labels
        if utt_labels is None:
            self.utt_labels = None
            self.num_utt_classes = None
        else:
            common.CHK_EQ(len(utt_labels), self.get_num_utts())
            self.utt_labels = numpy.asarray(utt_labels, dtype=labels_dtype)
            if num_utt_classes is None:
                self.num_utt_classes = numpy.max(self.utt_labels) + 1
            else:
                self.num_utt_classes = num_utt_classes
        # Then, process frame labels
        if frame_labels is None:
            if self.utt_labels is None:
                self.frame_labels = None
                self.num_frame_classes = None
            else:
                self.frame_labels = numpy.zeros(self.get_num_frames(),
                                                dtype=labels_dtype)

                def exp_labels(utt_name):
                    utt_label = self.get_utt_label_by_utt_name(utt_name)
                    num_frames = self.get_num_frames_by_utt_name(utt_name)
                    return numpy.repeat(utt_label, num_frames)

                fill_ary(self.frame_labels, self.get_utt_names(), exp_labels)
                self.num_frame_classes = self.get_num_utt_classes()
        else:
            # NOTE: hacky!
            common.CHK_EQ(len(frame_labels), self.data.shape[0] - 1)
            self.frame_labels = numpy.asarray(frame_labels, dtype=labels_dtype)
            if num_frame_classes is None:
                self.num_frame_classes = numpy.max(self.frame_labels) + 1
            else:
                self.num_frame_classes = num_frame_classes

    def set_utt_feats(self, utt_feats_dict=None, utt_feats_ary=None):
        """ When specifying `utt_feats_dict`, dummy features of all zeros will
        be used for utterances not present in `utt_feats_dict`.
        """
        assert not self.readonly
        self.utt_feats_dict = utt_feats_dict
        if utt_feats_ary is None:
            if utt_feats_dict is None:
                self.utt_feats_ary = None
            else:
                # NOTE: careful!
                utt_feats_dim = len(utt_feats_dict[utt_feats_dict.keys()[0]])
                num_frames = self.get_num_frames_by_utt_names(self.utt_names)
                dummy_feats = numpy.zeros(utt_feats_dim, dtype=self.get_dtype())
                self.utt_feats_ary = numpy.zeros((num_frames, utt_feats_dim),
                                                 dtype=self.get_dtype())

                def exp_feats(utt_name):
                    if utt_name in utt_feats_dict:
                        feats = utt_feats_dict[utt_name]
                    else:
                        io.log('...using dummy utt feats for {}'.format(utt_name))
                        feats = dummy_feats
                    utt_frames = self.get_num_frames_by_utt_name(utt_name)
                    shape = (utt_frames, utt_feats_dim)
                    return numpy.tile(feats, utt_frames).reshape(shape)
                fill_ary(self.utt_feats_ary, self.utt_names, exp_feats)
        else:
            self.utt_feats_ary = numpy.asarray(utt_feats_ary,
                                               dtype=self.get_dtype())

    def set_utt2group(self, utt2group):
        if utt2group is None:
            self.utt2group = None
            self.group2utts = None
        else:
            self.utt2group = utt2group
            self.group2utts = common.make_reverse_index(utt2group)

    def set_context(self, context, padding='zero', context_map=None,
                    dynamic_context=True):
        """ Context map is reset every time this function is called, unless
        a valid pre-computed context map is provided. Use with caution!
        """
        assert not self.readonly
        common.CHK_GE(context, 0)
        self.context = context
        common.CHK_VALS(padding, ['zero', 'replicate'])
        self.padding = padding
        self.dynamic_context = dynamic_context
        if context_map is None:
            self.__context_map_ary = None
        else:
            # Check that number of frames match
            # NOTE: this is a bit hacky, but will do for now
            common.CHK_EQ(context_map.shape[0], self.data.shape[0] - 1)
            # Check that feature dimensions match
            common.CHK_EQ(context_map.shape[1], self.get_window_size())
            self.__context_map_ary = context_map

    def set_seed(self, seed):
        self.seed = seed
        self.rng = numpy.random.RandomState(seed)

    def save(self, output_dir):
        """ Save dataset to the specified directory. If the directory doesn't
        currently exist, it will be created.
        """
        assert not self.readonly
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, 0755)
        # Data array
        numpy.save(os.path.join(output_dir, 'data.npy'), self.data)
        # Utterance names
        numpy.save(os.path.join(output_dir, 'utt_names.npy'), self.utt_names)
        # Utterance indices
        numpy.save(os.path.join(output_dir, 'start_indices.npy'),
                   self.start_indices)
        numpy.save(os.path.join(output_dir, 'end_indices.npy'),
                   self.end_indices)
        # Utterance labels
        if self.utt_labels is not None:
            numpy.save(os.path.join(output_dir, 'utt_labels.npy'),
                       self.utt_labels)
        # Frame labels
        if self.frame_labels is not None:
            numpy.save(os.path.join(output_dir, 'frame_labels.npy'),
                       self.frame_labels)
        # Utterance features
        if self.utt_feats_dict is not None:
            io.pickle_save(os.path.join(output_dir, 'utt_feats_dict.pkl'),
                           self.utt_feats_dict)
        # Utterance to group mapping
        if self.utt2group is not None:
            io.pickle_save(os.path.join(output_dir, 'utt2group.pkl'),
                           self.utt2group)
        # Editable properties
        io.json_save(os.path.join(output_dir, 'props.json'), self.get_props())

    ##############################
    ## Alternative constructors ##
    ##############################

    @classmethod
    def from_dir(cls, data_dir):
        """ Initialize from data directory. This mirrors the save() function.
        """
        objs = {}
        # Data array
        objs['data'] = numpy.load(os.path.join(data_dir, 'data.npy'))
        # Utterance names
        objs['utt_names'] = numpy.load(os.path.join(data_dir, 'utt_names.npy'))
        # Utterance indices
        objs['start_indices'] = \
            numpy.load(os.path.join(data_dir, 'start_indices.npy'))
        objs['end_indices'] = \
            numpy.load(os.path.join(data_dir, 'end_indices.npy'))
        # Utterance labels
        utt_labels_fname = os.path.join(data_dir, 'utt_labels.npy')
        if os.path.exists(utt_labels_fname):
            objs['utt_labels'] = numpy.load(utt_labels_fname)
        # Frame labels
        frame_labels_fname = os.path.join(data_dir, 'frame_labels.npy')
        if os.path.exists(frame_labels_fname):
            objs['frame_labels'] = numpy.load(frame_labels_fname)
        # Utterance features
        utt_feats_dict_fname = os.path.join(data_dir, 'utt_feats_dict.pkl')
        if os.path.exists(utt_feats_dict_fname):
            objs['utt_feats_dict'] = io.pickle_load(utt_feats_dict_fname)
        # Utterance to group mapping
        utt2group_fname = os.path.join(data_dir, 'utt2group.pkl')
        if os.path.exists(utt2group_fname):
            objs['utt2group'] = io.pickle_load(utt2group_fname)
        # Other properties
        props = io.json_load(os.path.join(data_dir, 'props.json'))
        # Construct object
        params = common.merge_dicts(objs, props)
        return cls(**params)

    @classmethod
    def from_kaldi(cls, scp, alipdf=None, num_pdfs=None, report_interval=None,
                   labels_dtype=numpy.int32, **kwargs):
        """ Initialize from Kaldi scp. This relies on pdnn.
        Note that the ark path in the scp should be absolute.
        """
        from io_func.kaldi_feat import KaldiReadIn
        kd = KaldiReadIn(scp)
        ali_dict = None if alipdf is None else io.dict_read(alipdf, lst=True)
        utt_names = []
        utt_mats = []
        frame_labels = None if alipdf is None else []
        end_indices = []
        frame_count = 0
        utts_read = 0
        while True:
            utt_name, utt_mat = kd.read_next_utt()
            if (utt_name, utt_mat) == ('', None):
                break
            utts_read += 1
            if report_interval is not None and utts_read % report_interval == 0:
                io.log('Read {} utterances'.format(utts_read))
            if alipdf is not None:
                if utt_name not in ali_dict:
                    io.log('**WARN** {} not in alipdf'.format(utt_name))
                    continue
                flabels = numpy.asarray(ali_dict[utt_name], dtype=labels_dtype)
                if utt_mat.shape[0] != len(flabels):
                    io.log('**WARN** {} - feats has {} rows but alipdf has {}'.format(
                        utt_name, utt_mat.shape[0], len(flabels)
                    ))
                    continue
                frame_labels.extend(flabels)
            utt_names.append(utt_name)
            utt_mats.append(utt_mat)
            frame_count += utt_mat.shape[0]
            end_indices.append(frame_count)
        utt_mats = numpy.vstack(utt_mats)
        return cls(utt_mats, utt_names, end_indices, frame_labels=frame_labels,
                   num_frame_classes=num_pdfs, labels_dtype=labels_dtype,
                   **kwargs)

    ##########################################################
    ## Methods that describe the properties of this dataset ##
    ##########################################################

    def get_labels_dtype(self):
        if self.frame_labels is None:
            return None
        else:
            return numpy.dtype(self.frame_labels.dtype)

    def get_dtype(self):
        return numpy.dtype(self.data.dtype)

    def get_window_size(self):
        """ Size of context window in terms of number of frames """
        return 1 + 2 * self.context

    def get_dim(self):
        return self.get_base_dim() * self.get_window_size() + \
            self.get_utt_feats_dim()

    def get_base_dim(self):
        return self.data.shape[1]

    def get_utt_feats_dim(self):
        if self.utt_feats_ary is None:
            return 0
        else:
            return self.utt_feats_ary.shape[1]

    def get_num_utts(self):
        return len(self.utt_names)

    def get_num_frames(self):
        return len(self.__frame_indices)

    def get_dummy_index(self):
        return self.dummy_index

    def get_utt_names(self):
        return self.utt_names

    def get_groups(self):
        return None if self.group2utts is None else self.group2utts.keys()

    def get_utt_names_by_group(self, group):
        assert self.group2utts is not None
        return self.group2utts[group]

    def get_utt_names_by_groups(self, groups):
        utt_names = []
        for group in groups:
            utt_names.extend(self.get_utt_names_by_group(group))
        return utt_names

    def get_group_by_utt_name(self, utt_name):
        assert self.utt2group is not None
        return self.utt2group[utt_name]

    def get_context(self):
        return self.context

    def get_padding(self):
        return self.padding

    def get_seed(self):
        return self.seed

    def get_num_utt_classes(self):
        return self.num_utt_classes

    def get_num_frame_classes(self):
        return self.num_frame_classes

    def get_props(self):
        props = OrderedDict()
        props['dummy_index'] = self.dummy_index
        props['num_utt_classes'] = self.num_utt_classes
        props['num_frame_classes'] = self.num_frame_classes
        props['context'] = self.context
        props['padding'] = self.padding
        props['seed'] = self.seed
        props['dtype'] = self.get_dtype().name
        return props

    ##########################
    ## Methods for indexing ##
    ##########################

    def get_shuffled_utt_indices(self):
        """ Return a 1D array containing shuffled utterance indices
        :attention: Returned array is immutable. DO NOT modify its content!
        """
        self.rng.shuffle(self.__utt_indices)
        return self.__utt_indices

    def get_shuffled_frame_indices(self):
        """ Return a 1D array containing shuffled frame indices
        :attention: Returned array is immutable. DO NOT modify its content!
        """
        self.rng.shuffle(self.__frame_indices)
        return self.__frame_indices

    def get_utt_index_by_utt_name(self, utt_name):
        return self.utt_to_index[utt_name]

    def get_utt_name_by_utt_index(self, utt_index):
        return self.utt_names[utt_index]

    def get_bounds_by_utt_name(self, utt_name):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_bounds_by_utt_index(utt_index)

    def get_bounds_by_utt_index(self, utt_index):
        return (self.start_indices[utt_index], self.end_indices[utt_index])

    def get_bounds_by_utt_names(self, utt_names):
        utt_indices = self.get_utt_indices_by_utt_names(utt_names)
        return self.get_bounds_by_utt_indices(utt_indices)

    def get_bounds_by_utt_indices(self, utt_indices):
        bounds = numpy.zeros((2, len(utt_indices)), dtype=numpy.int)
        bounds[0] = numpy.take(self.start_indices, utt_indices)
        bounds[1] = numpy.take(self.end_indices, utt_indices)
        return bounds.T

    def get_num_frames_by_utt_name(self, utt_name):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_num_frames_by_utt_index(utt_index)

    def get_num_frames_by_utt_index(self, utt_index):
        start_index, end_index = self.get_bounds_by_utt_index(utt_index)
        return end_index - start_index

    def get_num_frames_by_utt_names(self, utt_names):
        return accum_val(utt_names, self.get_num_frames_by_utt_name)

    def get_num_frames_by_utt_indices(self, utt_indices):
        return accum_val(utt_indices, self.get_num_frames_by_utt_index)

    def get_frame_indices(self):
        return self.get_frame_indices_by_utt_names(self.get_utt_names())

    def get_frame_indices_by_utt_name(self, utt_name):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_frame_indices_by_utt_index(utt_index)

    def get_frame_indices_by_utt_index(self, utt_index):
        start_index, end_index = self.get_bounds_by_utt_index(utt_index)
        return numpy.arange(start_index, end_index)

    def get_frame_indices_by_utt_names(self, utt_names):
        num_frames = self.get_num_frames_by_utt_names(utt_names)
        indices = numpy.zeros(num_frames, dtype=self.__frame_indices.dtype)
        fill_ary(indices, utt_names, self.get_frame_indices_by_utt_name)
        return indices

    def get_frame_indices_by_utt_indices(self, utt_indices):
        num_frames = self.get_num_frames_by_utt_indices(utt_indices)
        indices = numpy.zeros(num_frames, dtype=self.__frame_indices.dtype)
        fill_ary(indices, utt_indices, self.get_frame_indices_by_utt_index)
        return indices

    def get_utt_names_by_utt_indices(self, utt_indices):
        return numpy.take(self.utt_names, utt_indices)

    def get_utt_indices_by_utt_names(self, utt_names):
        indices = numpy.zeros(len(utt_names), dtype=self.__utt_indices.dtype)
        fill_vals(indices, utt_names, self.get_utt_index_by_utt_name)
        return indices

    def get_utt_index_by_frame_index(self, frame_index):
        common.CHK_RANGE(frame_index, 0, self.get_num_frames())
        return bisect_right(self.end_indices, frame_index)

    def get_utt_indices_by_frame_indices(self, frame_indices):
        indices = set()
        for frame_index in frame_indices:
            indices.add(self.get_utt_index_by_frame_index(frame_index))
        return indices

    def _utt_context_map(self, utt_index):
        start, end = self.get_bounds_by_utt_index(utt_index)
        num_frames = end - start
        # Initialize mapping contain dummy values
        indices = numpy.repeat(self.get_dummy_index(),
                               num_frames * self.get_window_size()).reshape(
                              ((num_frames, self.get_window_size())))
        # Tile with actual values
        vector = numpy.arange(start, end)
        for i in range(num_frames):
            self._fill_context(vector, i, row=indices[i])
        return indices

    def _fill_context(self, vector, offset, row=None):
        start, end = vector[0], vector[-1]
        if row is None:
            row = numpy.repeat(self.get_dummy_index(), self.get_window_size())
        if self.padding == 'replicate':
            row[:self.context] = start
            row[self.context+1:] = end
        # Tile with actual values
        left_pos = max(self.context - offset, 0)
        left_offset = max(offset - self.context, 0)
        right_pos = min(
            left_pos + end - start - left_offset + 1, self.get_window_size()
        )
        right_offset = left_offset + right_pos - left_pos
        row[left_pos:right_pos] = vector[left_offset:right_offset]
        return row

    def _context_map(self):
        """ This data structure describes how the data for each frame would
        look like after windowing. Specifically, it is a 2D array where the
        number of rows is the number of frames in this dataset and the number
        of columns is the feature dimension (post-expansion). The i-th row
        contains the frame indices in the context window that represents the
        i-th frame. The indices are sorted in ascending order, i.e. temporally.
        """
        # TODO: make this thread-safe!
        if self.__context_map_ary is None:
            assert not self.readonly
            self.__context_map_ary = \
                numpy.zeros((self.get_num_frames(),
                            self.get_window_size()), dtype=numpy.int)
            io.log("Building context map...")
            fill_ary(self.__context_map_ary, numpy.arange(self.get_num_utts()),
                     self._utt_context_map, report_interval=1000)
            io.log("Done!")
        return self.__context_map_ary

    def __get_context_indices_by_frame_index(self, frame_index):
        if self.dynamic_context:
            utt_index = self.get_utt_index_by_frame_index(frame_index)
            start, end = self.get_bounds_by_utt_index(utt_index)
            vector = numpy.arange(start, end)
            return self._fill_context(vector, frame_index - start)
        return self._context_map()[frame_index]

    def __get_context_indices_by_frame_indices(self, frame_indices):
        if self.dynamic_context:
            sorted_idx = numpy.argsort(frame_indices)
            n = len(frame_indices) * self.get_window_size()
            context_map = numpy.repeat(self.get_dummy_index(), n).reshape(
                ((len(frame_indices), self.get_window_size()))
            )
            utt_index, start, end, vector = -1, -1, -1, None
            for i in sorted_idx:
                frame_index = frame_indices[i]
                if frame_index >= end:
                    utt_index = self.get_utt_index_by_frame_index(frame_index)
                    start, end = self.get_bounds_by_utt_index(utt_index)
                    vector = numpy.arange(start, end)
                offset = frame_index - start
                self._fill_context(vector, offset, row=context_map[i])
            return context_map
        return numpy.take(self._context_map(), frame_indices, axis=0)

    ################################################
    ## Methods for accessing data in this dataset ##
    ################################################

    def get_all_data(self):
        return self.get_data_by_utt_names(self.get_utt_names())

    def get_data_by_utt_name(self, utt_name):
        frame_indices = self.get_frame_indices_by_utt_name(utt_name)
        return self.get_data_by_frame_indices(frame_indices)

    def get_data_by_utt_index(self, utt_index):
        frame_indices = self.get_frame_indices_by_utt_index(utt_index)
        return self.get_data_by_frame_indices(frame_indices)

    def get_data_by_utt_names(self, utt_names):
        frame_indices = self.get_frame_indices_by_utt_names(utt_names)
        return self.get_data_by_frame_indices(frame_indices)

    def get_data_by_utt_indices(self, utt_indices):
        frame_indices = self.get_frame_indices_by_utt_indices(utt_indices)
        return self.get_data_by_frame_indices(frame_indices)

    def get_data_by_frame_index(self, frame_index):
        indices = self.__get_context_indices_by_frame_index(frame_index)
        frame_feats = numpy.take(self.data, indices, axis=0).ravel()
        d, ud = self.get_dim(), self.get_utt_feats_dim()
        if ud == 0:
            return frame_feats
        else:
            data = numpy.zeros(d, dtype=self.get_dtype())
            data[:d-ud] = frame_feats
            data[d-ud:] = self.get_utt_feats_by_frame_index(frame_index)
            return data

    def get_data_by_frame_indices(self, frame_indices):
        indices = self.__get_context_indices_by_frame_indices(frame_indices)
        d, ud = self.get_dim(), self.get_utt_feats_dim()
        if ud == 0:
            data = numpy.take(self.data, indices, axis=0)
            return data.reshape((len(frame_indices), d))
        else:
            data = numpy.zeros((len(frame_indices), d), dtype=self.get_dtype())
            data[:, :d-ud] = numpy.take(self.data, indices, axis=0).reshape(
                (len(frame_indices), d-ud))
            data[:, d-ud:] = self.get_utt_feats_by_frame_indices(frame_indices)
            return data

    ##################################################
    ## Methods for accessing labels in this dataset ##
    ##################################################
    def get_utt_labels(self):
        return self.utt_labels

    def get_frame_labels(self):
        return self.frame_labels

    def get_utt_label_by_utt_name(self, utt_name):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_utt_label_by_utt_index(utt_index)

    def get_utt_label_by_utt_index(self, utt_index):
        assert self.utt_labels is not None
        return self.utt_labels[utt_index]

    def get_utt_labels_by_utt_names(self, utt_names):
        utt_indices = self.get_utt_indices_by_utt_names(utt_names)
        return self.get_utt_labels_by_utt_indices(utt_indices)

    def get_utt_labels_by_utt_indices(self, utt_indices):
        assert self.utt_labels is not None
        return numpy.take(self.utt_labels, utt_indices)

    def get_frame_label_by_frame_index(self, frame_index):
        assert self.frame_labels is not None
        return self.frame_labels[frame_index]

    def get_frame_labels_by_frame_indices(self, frame_indices):
        assert self.frame_labels is not None
        return numpy.take(self.frame_labels, frame_indices)

    def get_frame_labels_by_utt_name(self, utt_name):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_frame_labels_by_utt_index(utt_index)

    def get_frame_labels_by_utt_index(self, utt_index):
        assert self.frame_labels is not None
        start_index, end_index = self.get_bounds_by_utt_index(utt_index)
        return self.frame_labels[start_index:end_index]

    def get_frame_labels_by_utt_names(self, utt_names):
        num_frames = self.get_num_frames_by_utt_names(utt_names)
        labels = numpy.zeros(num_frames, dtype=self.get_labels_dtype())
        fill_ary(labels, utt_names, self.get_frame_labels_by_utt_name)
        return labels

    def get_frame_labels_by_utt_indices(self, utt_indices):
        num_frames = self.get_num_frames_by_utt_indices(utt_indices)
        labels = numpy.zeros(num_frames, dtype=self.get_labels_dtype())
        fill_ary(labels, utt_indices, self.get_frame_labels_by_utt_index)
        return labels

    ###############################################################
    ## Methods for accessing utterance features in this dataset. ##
    ###############################################################

    def get_utt_feats_dict(self):
        return self.utt_feats_dict

    def get_utt_feats_ary(self):
        return self.utt_feats_ary

    def get_utt_feats_by_utt_index(self, utt_index, expand=False):
        assert self.utt_feats_ary is not None
        start, end = self.get_bounds_by_utt_index(utt_index)
        if expand:
            return self.utt_feats_ary[start:end]
        else:
            return self.utt_feats_ary[start]

    def get_utt_feats_by_utt_name(self, utt_name, expand=False):
        utt_index = self.get_utt_index_by_utt_name(utt_name)
        return self.get_utt_feats_by_utt_index(utt_index, expand=expand)

    def get_utt_feats_by_utt_indices(self, utt_indices, expand=False):
        assert self.utt_feats_ary is not None
        if expand:
            num_frames = self.get_num_frames_by_utt_indices(utt_indices)
            feats = numpy.zeros((num_frames, self.get_utt_feats_dim()),
                                dtype=self.utt_feats_ary.dtype)

            def fill_fn(utt_index):
                return self.get_utt_feats_by_utt_index(utt_index, expand=True)

            fill_ary(feats, utt_indices, fill_fn)
        else:
            feats = numpy.zeros((len(utt_indices), self.get_utt_feats_dim()),
                                dtype=self.utt_feats_ary.dtype)

            def fill_fn(utt_index):
                return self.get_utt_feats_by_utt_index(utt_index, expand=False)

            fill_vals(feats, utt_indices, fill_fn)
        return feats

    def get_utt_feats_by_utt_names(self, utt_names, expand=False):
        utt_indices = self.get_utt_indices_by_utt_names(utt_names)
        return self.get_utt_feats_by_utt_indices(utt_indices, expand=expand)

    def get_utt_feats_by_frame_index(self, frame_index):
        assert self.utt_feats_ary is not None
        return self.utt_feats_ary[frame_index]

    def get_utt_feats_by_frame_indices(self, frame_indices):
        assert self.utt_feats_ary is not None
        return numpy.take(self.utt_feats_ary, frame_indices, axis=0)

    ############################################################
    ## Methods for trimming this dataset into a smaller set.  ##
    ## Our philosophy is to create a new object that provides ##
    ## a different view to the same underlying data, i.e. the ##
    ## data array is NOT copied over to the new object.       ##
    ############################################################

    def trim_by_utt_names(self, utt_names):
        """ Return a new dataset that contains only the specified utterances.
        """
        utt_names = numpy.asarray(utt_names, dtype=numpy.str)
        objs = {}
        objs['data'] = self.data
        objs['context_map'] = self._context_map()
        objs['utt_names'] = utt_names
        bounds = self.get_bounds_by_utt_names(utt_names)
        objs['start_indices'] = bounds[:, 0]
        objs['end_indices'] = bounds[:, 1]
        if self.utt_labels is not None:
            objs['utt_labels'] = self.get_utt_labels_by_utt_names(utt_names)
        if self.frame_labels is not None:
            objs['frame_labels'] = self.frame_labels
        if self.utt_feats_ary is not None:
            objs['utt_feats_ary'] = self.utt_feats_ary
        if self.utt2group is not None:
            utt_groups = map(self.get_group_by_utt_name, utt_names)
            objs['utt2group'] = common.make_index(utt_names, utt_groups)
        props = self.get_props()
        props['readonly'] = True
        # Construct object
        params = common.merge_dicts(objs, props)
        return TemporalData(**params)

    def trim_by_groups(self, groups):
        utt_names = self.get_utt_names_by_groups(groups)
        return self.trim_by_utt_names(utt_names)

    def trim_by_group(self, group):
        utt_names = self.get_utt_names_by_group(group)
        return self.trim_by_utt_names(utt_names)

    def leave_one_group_out_partitions(self):
        """ Performs leave-one-group-out partitioning on this dataset.
        Returns X partitions where X is the number of groups. Each partition
        is a tuple containing two items. The first is a list containing the
        withheld group names. The second is the left out group name.
        """
        groups = self.get_groups()
        assert groups is not None
        common.CHK_GE(len(groups), 2)
        partitions = []
        for i in numpy.arange(len(groups)):
            withhelds = []
            for j in numpy.arange(len(groups)):
                if j != i:
                    withhelds.append(groups[j])
            partitions.append((withhelds, groups[i]))
        return partitions


class MultiLabelTemporalData(TemporalData):
    """
    Extension of `TemporalData`, where frames/utterances have multiple labels.
    TODO: allow multiple label types.
    """
    def set_labels(self, utt_labels=None, num_utt_classes=None,
                   frame_labels=None, num_frame_classes=None,
                   labels_dtype=numpy.int32):
        """
        Similar to the `TemporalData` version, with these differences:

        * `utt_labels`, `num_utt_classes` are None (default) or lists
        * `frame_labels`, `num_frame_classes` are None (default) or lists
        * `labels_dtype` can be a single type (applies to all labels) or a list
        """
        assert not self.readonly
        # First, process utterance labels
        if utt_labels is None:
            self.utt_labels = None
            self.num_utt_classes = None
        else:
            common.CHK_EQ(type(utt_labels), list)
            if type(labels_dtype) == list:
                common.CHK_EQ(len(utt_labels), len(labels_dtype))
            else:
                labels_dtype = [labels_dtype] * len(utt_labels)
            self.utt_labels = []
            self.num_utt_classes = []
            for i in range(len(utt_labels)):
                common.CHK_EQ(len(utt_labels[i]), self.get_num_utts())
                self.utt_labels.append(numpy.asarray(utt_labels[i],
                                                     dtype=labels_dtype[i]))
                if num_utt_classes is None:
                    self.num_utt_classes.append(numpy.max(self.utt_labels[i]) + 1)
                else:
                    common.CHK_EQ(type(num_utt_classes), list)
                    common.CHK_EQ(len(num_utt_classes), len(utt_labels))
                    self.num_utt_classes.append(num_utt_classes[i])
        # Then, process frame labels
        if frame_labels is None:
            if self.utt_labels is None:
                self.frame_labels = None
                self.num_frame_classes = None
            else:
                self.frame_labels = []
                self.num_frame_classes = []
                for i in range(len(self.utt_labels)):
                    self.frame_labels.append(numpy.zeros(self.get_num_frames(),
                                                         dtype=labels_dtype[i]))

                    def exp_labels(utt_name):
                        utt_label = self.get_utt_label_by_utt_name(utt_name)[i]
                        num_frames = self.get_num_frames_by_utt_name(utt_name)
                        return numpy.repeat(utt_label, num_frames)

                    fill_ary(self.frame_labels[i], self.get_utt_names(), exp_labels)
                    self.num_frame_classes.append(self.get_num_utt_classes()[i])
        else:
            # NOTE: hacky!
            common.CHK_EQ(type(frame_labels), list)
            if type(labels_dtype) == list:
                common.CHK_EQ(len(frame_labels), len(labels_dtype))
            else:
                labels_dtype = [labels_dtype] * len(frame_labels)
            self.frame_labels = []
            self.num_frame_classes = []
            for i in range(len(frame_labels)):
                self.frame_labels.append(numpy.asarray(frame_labels[i],
                                                       dtype=labels_dtype[i]))
                if num_frame_classes is None:
                    self.num_frame_classes.append(numpy.max(self.frame_labels[i]) + 1)
                else:
                    common.CHK_EQ(type(num_frame_classes), list)
                    common.CHK_EQ(len(num_frame_classes), len(frame_labels))
                    self.num_frame_classes.append(num_frame_classes[i])

    def save(self, output_dir):
        raise NotImplementedError('Not yet implemented!')

    ##############################
    ## Alternative constructors ##
    ##############################

    @classmethod
    def from_dir(cls, data_dir):
        raise NotImplementedError('Not yet implemented!')

    @classmethod
    def from_kaldi(cls, scp, alipdfs=None, num_pdfs=None, report_interval=None,
                   labels_dtype=numpy.int32, **kwargs):
        """ Initialize from Kaldi scp. This relies on pdnn.
        Note that the ark path in the scp should be absolute.
        """
        from io_func.kaldi_feat import KaldiReadIn
        kd = KaldiReadIn(scp)
        ali_dicts = None
        utt_names = []
        utt_mats = []
        frame_labels = None
        if alipdfs is not None:
            common.CHK_EQ(type(alipdfs), list)
            if type(labels_dtype) == list:
                common.CHK_EQ(len(alipdfs), len(labels_dtype))
            else:
                labels_dtype = [labels_dtype] * len(alipdfs)
            ali_dicts = [io.dict_read(ap, lst=True) for ap in alipdfs]
            frame_labels = [[] for _ in alipdfs]
        end_indices = []
        frame_count = 0
        utts_read = 0
        while True:
            utt_name, utt_mat = kd.read_next_utt()
            if (utt_name, utt_mat) == ('', None):
                break
            utts_read += 1
            if report_interval is not None and utts_read % report_interval == 0:
                io.log('Read {} utterances'.format(utts_read))
            if alipdfs is not None:
                if any([utt_name not in ad for ad in ali_dicts]):
                    io.log('**WARN** {} not in alipdfs'.format(utt_name))
                    continue
                flabels = []
                for i in range(len(alipdfs)):
                    flabels.append(numpy.asarray(ali_dicts[i][utt_name],
                                                 dtype=labels_dtype[i]))
                if any([utt_mat.shape[0] != len(fl) for fl in flabels]):
                    io.log('**WARN** {} - feats has {} rows but alipdfs has {}'.format(
                        utt_name, utt_mat.shape[0], [len(fl) for fl in flabels]
                    ))
                    continue
                for i in range(len(alipdfs)):
                    frame_labels[i].extend(flabels[i])
            utt_names.append(utt_name)
            utt_mats.append(utt_mat)
            frame_count += utt_mat.shape[0]
            end_indices.append(frame_count)
        utt_mats = numpy.vstack(utt_mats)
        return cls(utt_mats, utt_names, end_indices, frame_labels=frame_labels,
                   num_frame_classes=num_pdfs, labels_dtype=labels_dtype,
                   **kwargs)

    ##########################################################
    ## Methods that describe the properties of this dataset ##
    ##########################################################

    def get_labels_dtype(self):
        if self.frame_labels is None:
            return None
        else:
            labels_dtype = []
            for i in range(len(self.frame_labels)):
                labels_dtype.append(numpy.dtype(self.frame_labels[i].dtype))
            return labels_dtype

    ##################################################
    ## Methods for accessing labels in this dataset ##
    ##################################################

    def get_utt_label_by_utt_index(self, utt_index):
        assert self.utt_labels is not None
        return [ul[utt_index] for ul in self.utt_labels]

    def get_utt_labels_by_utt_indices(self, utt_indices):
        assert self.utt_labels is not None
        return [numpy.take(ul, utt_indices) for ul in self.utt_labels]

    def get_frame_label_by_frame_index(self, frame_index):
        assert self.frame_labels is not None
        return [fl[frame_index] for fl in self.frame_labels]

    def get_frame_labels_by_frame_indices(self, frame_indices):
        assert self.frame_labels is not None
        return [numpy.take(fl, frame_indices) for fl in self.frame_labels]

    def get_frame_labels_by_utt_index(self, utt_index):
        assert self.frame_labels is not None
        start_index, end_index = self.get_bounds_by_utt_index(utt_index)
        return [fl[start_index:end_index] for fl in self.frame_labels]

    def get_frame_labels_by_utt_names(self, utt_names):
        def fn(i):
            return lambda utt_name: self.get_frame_labels_by_utt_name(utt_name)[i]
        num_frames = self.get_num_frames_by_utt_names(utt_names)
        labels = []
        for i in range(len(self.frame_labels)):
            labels.append(numpy.zeros(num_frames, dtype=self.get_labels_dtype()[i]))
            fill_ary(labels[i], utt_names, fn(i))
        return labels

    def get_frame_labels_by_utt_indices(self, utt_indices):
        def fn(i):
            return lambda utt_index: self.get_frame_labels_by_utt_index(utt_index)[i]
        num_frames = self.get_num_frames_by_utt_indices(utt_indices)
        labels = []
        for i in range(len(self.frame_labels)):
            labels.append(numpy.zeros(num_frames, dtype=self.get_labels_dtype()[i]))
            fill_ary(labels[i], utt_indices, fn(i))
        return labels

    ############################################################
    ## Methods for trimming this dataset into a smaller set.  ##
    ## Our philosophy is to create a new object that provides ##
    ## a different view to the same underlying data, i.e. the ##
    ## data array is NOT copied over to the new object.       ##
    ############################################################

    def trim_by_utt_names(self, utt_names):
        raise NotImplementedError('Not yet implemented!')


class BufferedData(object):
    """
    This abstract class defines the most basic functions that any data buffering
    class must implement. Its usage will revolve around these functions.
    """
    def dataset(self):
        """ Expose underlying dataset to users
        """
        raise NotImplementedError()

    def read_next_chunk(self):
        """ Return the next chunk of data, or None if no more data can be read.
        """
        raise NotImplementedError()

    def get_delay(self):
        """ Return the delay between frames and labels.
        """
        return 0

    def reset(self):
        """ Reset data buffering.
        """
        raise NotImplementedError()

    def get_progress(self):
        """ Return a string that shows buffering progress.
        """
        return "N/A"


class BufferedFrameData(BufferedData):
    """
    This class is a wrapper around TemporalData to support reading frame-level
    data in chunks. This is useful for GPU training, where we can only send
    a relatively small chunk of data to the GPU at any given time.
    """
    def __init__(self, dataset, chunk_size='800m', excluded_frame_labels=[]):
        """
        :type dataset: chai.data.temporal.TemporalData
        :param dataset: The dataset to buffer on

        :type chunk_size: str
        :param chunk_size: String denoting the size of the chunk in memory.
            Acceptable format is a number followed by 'm' or 'M' for megabytes
            or 'g' or 'G' for gigabytes. For example: '600m', '1.2g'.
        """
        self.__dataset = dataset
        self.frame_indices = self.dataset().get_frame_indices()
        self.set_chunk_size(chunk_size)

    def dataset(self):
        return self.__dataset

    def get_chunk_size(self):
        return self.chunk_size

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size
        # Each time chunk_size is changed, reset the partitions
        self.reset()

    def reset(self):
        self.current_index = 0
        self.dataset().rng.shuffle(self.frame_indices)
        num_items = get_num_items(self.get_chunk_size(),
                                  self.dataset().get_dtype())
        num_frames = num_items / self.dataset().get_dim()
        split_indices = numpy.arange(num_frames, len(self.frame_indices), num_frames)
        self.partitions = numpy.split(self.frame_indices, split_indices)

    def max_partition_size(self):
        """ Return the maximum partition size in frames """
        return numpy.max(map(lambda x: len(x), self.partitions))

    def read_next_chunk(self):
        """ Return a tuple containing the following items in this order:
            (1) Array containing frame features.
            (2) Array containing frame labels. Will be set to None if
                dataset does not have frame labels.
            (3) Array containing the frame indices that were used.

        If this function returns None, it means there is no more chunk to be
        read. After this, calling the function will repeat the partitioning,
        but with a different order of frame indices.
        """
        if self.current_index >= len(self.partitions):
            self.reset()
            return None
        else:
            frames = self.partitions[self.current_index]
            feats = self.dataset().get_data_by_frame_indices(frames)
            labels = None if self.dataset().get_frame_labels() is None else \
                    self.dataset().get_frame_labels_by_frame_indices(frames)
            self.current_index += 1
            return (feats, labels, frames)

    def get_progress(self):
        return '{}/{}'.format(self.current_index, len(self.partitions))


class BufferedTemporalData(BufferedFrameData):
    """
    This class is for pdnn only. For keras, use `BufferedFrameData` instead!
    """
    def make_shared(self, use_labels=True, x_name='x', y_name='y'):
        """ Return a shared dataset intended for training with Theano. The
        numpy array is as large as the largest chunk in this dataset. The
        shared Theano array references the numpy array. If use_labels is True,
        create shared data for both features and frame labels. Otherwise, only
        create shared data for features. Also return the array size in frames.

        If use_labels is True, return (x, shared_x, y, shared_y, max_frames).
        If use_labels is False, return (x, shared_x, max_frames).
        """
        import theano
        max_frames = self.max_partition_size()
        x = numpy.zeros((max_frames, self.dataset().get_dim()),
                        dtype=self.dataset().get_dtype())
        shared_x = theano.shared(x, name=x_name, borrow=True)
        if use_labels:
            if type(self.dataset().get_labels_dtype()) == list:
                ys = []
                shared_ys = []
                for i in range(len(self.dataset().get_labels_dtype())):
                    ltype = self.dataset().get_labels_dtype()[i]
                    ys.append(numpy.zeros(max_frames, dtype=ltype))
                    shared_ys.append(theano.shared(
                        ys[i], name='{}_{}'.format(y_name, i), borrow=True
                    ))
                return (x, shared_x, ys, shared_ys, max_frames)
            else:
                y = numpy.zeros(max_frames, dtype=self.dataset().get_labels_dtype())
                shared_y = theano.shared(y, name=y_name, borrow=True)
                return (x, shared_x, y, shared_y, max_frames)
        else:
            return (x, shared_x, max_frames)


class BufferedUttData(BufferedData):
    """
    This class is a wrapper around `TemporalData` to support reading utt-level
    data in chunks. This is useful for RNN training, intended to be used with
    Keras. It'll have less low-level functionality than `BufferedTemporalData`.
    Note that properties of this class cannot be changed after initialization,
    unlike `chunk_size` in `BufferedTemporalData`.
    """
    def __init__(self, dataset, max_frames=None, nutts=10, nframes=None, delay=0,
                 feat_padding=0.0, label_padding=0, length_bins=50, shuffle=True):
        """
        :type dataset: chai.data.temporal.TemporalData
        :param dataset: The dataset to buffer on

        :type max_frames: int
        :param max_frames: Only consider utterances shorter than or equal to
            this. If None, use all utterances.

        :type nutts: int
        :param nutts: Number of utterances per minibatch.

        :type nframes: int
        :param nframes: If specified, each utterance will be chopped into
            minibatches of this length or less. If None, use whole utterance.

        :type delay: int
        :param delay: Delay output by this many frames. This means the last
            frame of the feature matrix is repeated `delay` times, and the
            label vector starts `delay` frames after the first frame.

        :type feat_padding: float
        :param feat_padding: Value used for feature padding.

        :type label_padding: int
        :param label_padding: Value used for label padding.

        :type length_bins: int
        :param length_bins: Used to ensure that utterances in the same batch
            have roughly similar length. The more bins there are, the closer
            in length the utterances are, but randomness is decreased. If set
            to 0, utterances will not be arranged by length at all.

        :type shuffle: bool
        :param shuffle: Whether or not to shuffle utterance orders.
        """
        self.__dataset = dataset
        self._max_frames = max_frames
        self._nutts = nutts
        self._nframes = nframes
        self._delay = delay
        self._feat_padding = feat_padding
        self._label_padding = label_padding
        self._init_length_bins(length_bins)
        self._shuffle = shuffle
        self.reset()

    def _init_length_bins(self, num_bins):
        if self._max_frames is None:
            utts = self.dataset().get_utt_names()
        else:
            utts = [u for u in self.dataset().get_utt_names() if \
                    self.dataset().get_num_frames_by_utt_name(u) <= self._max_frames]
            io.log('Removed {} utts with length > {}'.format(
                len(self.dataset().get_utt_names())-len(utts), self._max_frames
            ))
        idx = self.dataset().get_utt_indices_by_utt_names(utts)
        lengths = [self.dataset().get_num_frames_by_utt_name(u) for u in utts]
        sorted_idx = idx[numpy.argsort(lengths)]
        self._max_frames = numpy.max(lengths)
        io.log('Max utt length: {}'.format(self._max_frames))
        self._length_bins = [idx] if num_bins <= 0 else \
            partition(sorted_idx, 1.0 / num_bins)

    def dataset(self):
        """ Expose underlying dataset to users """
        return self.__dataset

    def reset(self):
        self.current_index = 0
        utt_indices = []
        for i in range(len(self._length_bins)):
            if self._shuffle:
                self.dataset().rng.shuffle(self._length_bins[i])
            utt_indices.extend(self._length_bins[i])
        utt_indices = numpy.asarray(utt_indices)
        split_indices = self._split_indices(utt_indices)
        self.partitions = numpy.split(utt_indices, split_indices)
        if self._shuffle:
            self.dataset().rng.shuffle(self.partitions)

    def _split_indices(self, utt_indices):
        return numpy.arange(self._nutts, len(utt_indices), self._nutts)

    def read_next_chunk(self):
        """ Return a tuple containing the following items in this order:
            (1) Frame features for utterances, as a list of 3D arrays.
                * If `nframes` is None, this will contain one 3D array of shape
                  (nutt, max_utt_length, dim)
                * Otherwise, this will be a list of 3D arrays of shape
                  (nutt, [1, nframes], dim)
            (2) Frame labels for utterances. If dataset does not have frame
                labels, this will be None. If not, return a list of 2D arrays.
                * If `nframes` is None, this will contain one 2D array of shape
                  (nutt, max_utt_length)
                * Otherwise, this will be a list of 2D arrays of shape
                  (nutt, [1, nframes])
            (3) End of batch indicators, as a list of 1D arrays.
                * If `nframes` is None, this will contain one 1D array of shape
                  (nutt,)
                * Otherwise, this will be a list of 1D arrays of shape
                  (nutt,)
                The array cell contains the number of frames that are valid
                within the utterance. A cell value of 0 means the utterance has
                no valid element, i.e. it has ended in a previous batch.
            (4) Array containing the utterance indices that were used.

        If this function returns None, it means there is no more chunk to be
        read. After this, calling the function will repeat the partitioning,
        but with a different order of utterance indices.
        """
        if self.current_index >= len(self.partitions):
            self.reset()
            return None
        else:
            utts = self.partitions[self.current_index]
            durs = [self.dataset().get_num_frames_by_utt_index(u) for u in utts]
            feats = [self.dataset().get_data_by_utt_index(u) for u in utts]
            labels = None if self.dataset().get_frame_labels() is None else \
                [self.dataset().get_frame_labels_by_utt_index(u) for u in utts]
            self.current_index += 1
            return self._make_chunk(feats, labels, utts, durs)

    def _make_chunk(self, feats, labels, utts, durs):
        max_dur = numpy.max(durs)
        step = max_dur
        if self._nframes is not None:
            step = min(self._nframes, step)
        start = 0
        markers = [0] * len(utts)
        c_feats, c_labels, c_eob = [], [], []
        while start < max_dur:
            dur = min(step, max_dur - start)
            # Initialize containers
            tmp_feats, tmp_labels, tmp_eob = \
                self._init_containers(len(utts), dur, labels)
            # Fill values
            for i in range(len(utts)):
                if markers[i] < durs[i]:
                    u_step = min(step, durs[i] - markers[i])
                    tmp_feats[i][0:u_step] = feats[i][start:start+u_step]
                    if self._delay > 0:
                        frames_left = durs[i] - (start + u_step)
                        spill = min(frames_left, self._delay)
                        repeat = self._delay - spill
                        tmp_feats[i][u_step:u_step+spill] = \
                            feats[i][start+u_step:start+u_step+spill]
                        tmp_feats[i][u_step+spill:u_step+spill+repeat] = \
                            feats[i][start+u_step+spill-1]
                    if labels is not None:
                        if type(self.dataset().get_frame_labels()) == list:
                            for j in range(len(self.dataset().get_frame_labels())):
                                tmp_labels[j][i][self._delay:self._delay+u_step] = \
                                    labels[i][j][start:start+u_step]
                        else:
                            tmp_labels[i][self._delay:self._delay+u_step] = \
                                labels[i][start:start+u_step]
                    tmp_eob[i] = u_step + self._delay
                    markers[i] += u_step
            # Save containers
            c_feats.append(tmp_feats)
            c_labels.append(tmp_labels)
            c_eob.append(tmp_eob)
            start += step
        return (c_feats, None if labels is None else c_labels, c_eob, utts)

    def _init_containers(self, nutts, dur, labels):
        tmp_feats = numpy.empty(
            (nutts, dur + self._delay, self.dataset().get_dim()),
            dtype=self.dataset().data.dtype
        )
        tmp_feats.fill(self._feat_padding)
        tmp_labels = None
        if labels is not None:
            if type(self.dataset().get_frame_labels()) == list:
                tmp_labels = []
                for i in range(len(self.dataset().get_frame_labels())):
                    tmp_labels.append(numpy.empty(
                        (nutts, dur + self._delay), self.dataset().get_labels_dtype()[i]
                    ))
                    tmp_labels[i].fill(self._label_padding)
            else:
                tmp_labels = numpy.empty(
                    (nutts, dur + self._delay), dtype=self.dataset().get_labels_dtype()
                )
                tmp_labels.fill(self._label_padding)
        tmp_eob = numpy.zeros(nutts, dtype=numpy.int)
        return (tmp_feats, tmp_labels, tmp_eob)

    def get_delay(self):
        return self._delay

    def get_progress(self):
        return '{}/{}'.format(self.current_index, len(self.partitions))


class BufferedVarUttData(BufferedUttData):
    """
    Like `BufferedUttData`, but the number of utterances returned per chunk
    differs depending on the maximum utterance length in that chunk. The
    total number of elements will never exceed `nutts` * `max_frames`.
    The purpose of this class is to reduce the number of GPU data transfers.
    """
    def _split_indices(self, utt_indices):
        cap = self._nutts * self._max_frames
        num_frames = self.dataset().get_num_frames_by_utt_index
        split_indices = []
        while True:
            start = 0 if len(split_indices) == 0 else split_indices[-1]
            max_frames = num_frames(utt_indices[start])
            end = start + 1
            while end < len(utt_indices) and (end - start) * max_frames <= cap:
                max_frames = max(max_frames, num_frames(utt_indices[end]))
                end += 1
            if end == len(utt_indices):
                break
            common.CHK_GT(end - 1, start)
            split_indices.append(end - 1)
        return split_indices


class AugmentedBufferedUttData(BufferedUttData):
    """
    Like `BufferedUttData`, with the labels augmented with model output.
    """
    def __init__(self, dataset, model, **kwargs):
        assert isinstance(dataset, MultiLabelTemporalData)
        super(AugmentedBufferedUttData, self).__init__(dataset, **kwargs)
        self.model = model

    def read_next_chunk(self):
        X = super(AugmentedBufferedUttData, self).read_next_chunk()
        if X is None:
            return None
        xs, ys, eobs, utts = X
        if ys is not None:
            for x, y in zip(xs, ys):
                common.CHK_EQ(type(y), list)    # dataset must be multi-labeled
                preds = self.model.predict(x, batch_size=len(utts), verbose=0)
                if type(preds) != list:
                    preds = [preds]
                y.extend(preds)
        return (xs, ys, eobs, utts)


class AugmentedBufferedVarUttData(BufferedVarUttData, AugmentedBufferedUttData):
    """
    Like `BufferedVarUttData`, with the labels augmented with model output.
    """
    pass

