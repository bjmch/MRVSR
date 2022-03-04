# -*- coding: utf-8 -*-

import numpy as np
import os
import re


class SequenceFilepathParser:
    """ ** Sequence Filepath Parser Class**

    Parser object used to parse and load paths of sequences of files
    """

    def __init__(self, *directories, n_frames=None, matching_pattern=None, starting_number=0, regex=None, excluding_pattern=None,
                 ignore_case=True, use_random_starting_number=False,
                 jump_size=1, use_random_files=False, max_number_of_sequences=None):
        """ **Initialization function**

        Args:
            *directories (string): Path to directory or directories containing the file sequences to read.
            n_frames (int, None): Maximum size of the sequences which will be read. If the sequences are longer, they will be truncated. (default None).
            matching_pattern (string or list): Pattern which the files in the directory should match to be read.(default None).
            starting_number (int): Number where the file sequences start.(default 0).
            regex (string): Regex to indicate how the sequences are written. If both suffixe and regex are provided, suffixe will be used. (default None).
            excluding_pattern (string, list[str]): Pattern which the files in the directory should not match to be read.(default None).
            ignore_case (bool): Bool to ignore case or not in the research of the matching and excluding pattern
            use_random_starting_number (bool): If True, the sub-sequence is randomly placed in the sequence. Otherwise it starts at the first path.
            jump_size (int): Image temporal downsampling : to use only one file every jump_size
            use_random_files (bool): If True, the sequences are random picks of paths
            max_number_of_sequences (int, None): If set, truncates the dataset using only that many elements, in alphabetical order.
        """

        self.matching_pattern = matching_pattern  # matching pattern for files or directories
        self.excluding_pattern = excluding_pattern  # excluding pattern for files or directories
        self.ignore_case = ignore_case  # bool to ignore case for matching or excluding pattern

        self.n_frames = n_frames  # number of frames to return. If None, returns the whole sequence
        self.use_random_frames = use_random_files
        self.use_random_starting_number = use_random_starting_number
        self.starting_number = starting_number  # to skip first frames
        self.jump_size = jump_size
        self.regex = regex  # variable part regular expression. for example "[0-9]+". Will be sorted by numpy.
        assert (self.n_frames or not self.use_random_starting_number), ValueError('use_random_starting_number cannot be used when n_frames is None')

        self._directories = directories
        self.path_list = list()
        self.seq_files = dict()
        self.seq_length = dict()

        self.max_number_of_sequences = max_number_of_sequences

        self.re_flags = re.IGNORECASE if self.ignore_case else 0

        self._compute_path_lists()

    def __getitem__(self, index, starting_number=None, return_starting_number=False):
        """ **Indexation operator function**

        Returns the images sequence corresponding the index provided.

        Args:
            index (int): Index of the images sequence to be read.

        Returns:
            list of numpy array: Numpy arrays containing the images sequence read

        """

        assert isinstance(index, int), "The index must be integer."
        assert (index > -1) and (index < self.__len__()), f"The index must be in range(0, {self.__len__()})."

        # Current sequence id (path + starting string)
        seq_id = self.path_list[index]
        n_files = self.seq_length[seq_id]

        # Number of frames and starting number
        if self.n_frames is None:
            if starting_number is None:
                starting_number = self.starting_number
            assert (n_files > starting_number), ValueError(f'starting_number ({starting_number}) greater than number of files ({n_files}) in sequence {seq_id}')
            n_frames = n_files - starting_number
        else:
            n_frames = self.n_frames
            # First image number (in filename)
            if starting_number is None:
                if not self.use_random_starting_number:
                    starting_number = self.starting_number
                else:
                    starting_number = np.random.randint(self.starting_number, n_files - self.jump_size * (n_frames - 1))
            assert (n_files >= starting_number + n_frames), ValueError(f'starting_number ({starting_number}) + n_framse ({n_frames}) greater than number of files ({n_files}) in sequence {seq_id}')

        if self.use_random_frames:
            def get_file_number(_: int):
                return np.random.randint(starting_number, n_files)
        else:
            def get_file_number(k: int):
                return starting_number + k * self.jump_size

        # Read images : generate a list of frames
        file_paths = list()
        for i in range(n_frames):
            file_paths.append(self.seq_files[seq_id][get_file_number(i)])

        # print('reader', id(self), 'seq, frame:', index, starting_number)
        if not return_starting_number:
            return file_paths
        else:
            return file_paths, starting_number

    def check_pattern(self, file_path, patterns):
        if isinstance(patterns, list):
            for pattern in patterns:
                if re.search(pattern, file_path, flags=self.re_flags):
                    return True
        else:
            if re.search(patterns, file_path, flags=self.re_flags):
                return True
        return False

    def check_patterns(self, file_path):
        return (not self.matching_pattern or self.check_pattern(file_path, self.matching_pattern)) \
               and (not self.excluding_pattern or not self.check_pattern(file_path, self.excluding_pattern))

    def _compute_path_lists(self):
        """ **Images sequences path computation function**

        Private function to compute the different file sequence paths.

        """

        directories = self._directories

        if self.n_frames is not None:
            n_frames_min = self.n_frames + self.starting_number + (self.n_frames - 1) * (self.jump_size - 1)
        else:
            n_frames_min = 1

        for directory in directories:
            for path, dirs, files in os.walk(directory, followlinks=True):
                # If enough matching file here, add current path and its matching files
                file_list = list()
                for file in files:
                    file_path = os.path.join(path, file)
                    if self.check_patterns(file_path):
                        file_list.append(file_path)
                if len(file_list) >= n_frames_min:
                    self.path_list.append(path)
                    self.seq_files[path] = sorted(file_list)
                    self.seq_length[path] = len(self.seq_files[path])

        # Sort sequence paths (to ensure order consistency with other similar file trees)
        self.path_list.sort()

        # Delete not needed file lists
        if self.max_number_of_sequences:
            for p in set(self.path_list[self.max_number_of_sequences:]):
                if p not in self.path_list[:self.max_number_of_sequences] and p in self.seq_files:
                    self.seq_files.pop(p)
                    self.seq_length.pop(p)
            self.path_list = self.path_list[:self.max_number_of_sequences]

        if len(self.path_list) == 0:
            raise ValueError(f'There is no matching sequence ({directories}, {self.matching_pattern}) with enough frames ({self.starting_number}, {self.n_frames}, {self.jump_size})')

    def __len__(self):
        """ **Length operator function**

        Returns the number of sequences which can be read

        Returns:
            int: Number of sequences which can be read

        """

        return len(self.path_list)

    def __str__(self):
        """ **Conversion to string function**

        This function transforms a data reader into a string which can be printed directly by typing the name of the reader in command terminal.
        This string describes all the information of the data reader..

        Returns:
            string: Text containing all the information of the data reader which can be printed.

        """

        infos = "------------------------\n"
        infos += " SequenceFilepathParser \n"
        infos += "------------------------\n"
        infos += "Source directories  :  \n"
        for directory in self._directories:
            infos += f"\t -{directory}\n"
        infos += f"n_sequences         : {len(self)}\n"
        infos += f"n_frames            : {self.n_frames}\n"
        infos += f"starting_number     : {self.starting_number}\n"
        infos += f"matching pattern    : {self.matching_pattern}\n"
        infos += f"excluding pattern   : {self.excluding_pattern}\n"
        infos += "-----------------------------------"

        return infos

    def __repr__(self):
        return self.__str__()
