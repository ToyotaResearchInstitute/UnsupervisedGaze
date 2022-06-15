"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from collections import OrderedDict
import random

class TypedOrderedDict(OrderedDict):
    def __init__(self, key_type, value_type):
        super(TypedOrderedDict, self).__init__()
        self.key_type = key_type
        self.value_type = value_type

    def insert_list_of_strings(self, list_of_strings):
        assert (len(list_of_strings) % 2) == 0
        for i in range(0,len(list_of_strings),2):
            key = self.key_type(list_of_strings[i])
            value = self.value_type(list_of_strings[i+1])
            self[key] = value

class MultiDict():
    def __init__(self, level_order):
        self.level_order = level_order  # Should be list of levels
        self.level = level_order[0]  # Should be the name of this level
        self.data = {}

    def __setitem__(self, keys, value):
        level_key = keys[self.level]
        if len(self.level_order) == 1:
            self.data[level_key] = value
        else:
            if level_key not in self.data:
                self.data[level_key] = MultiDict(self.level_order[1:])
            self.data[level_key][keys] = value

    def __getitem__(self, keys):
        if self.level not in keys:
            return self
        level_key = keys[self.level]
        if len(self.level_order) == 1:
            return self.data[level_key]
        else:
            return self.data[level_key][keys]

    def __contains__(self, keys):
        level_key = keys[self.level]
        if level_key in self.data:
            if len(self.level_order) == 1:
                return True
            else:
                return keys in self.data[level_key]
        else:
            return False

    def __delitem__(self, keys):
        level_key = keys[self.level]
        if len(self.level_order) == 1:
            del self.data[level_key]
        else:
            del self.data[level_key][keys]

    def __len__(self):
        return len(list(self.keys()))

    def keys(self, stop_level=None):
        if stop_level is None:
            stop_level = self.level_order[-1]
        if self.level == stop_level:
            for key in self.data.keys():
                yield {self.level: key} 
            return
        for level_key, sub_dict in self.data.items():
            for sub_keys in sub_dict.keys(stop_level):
                sub_keys[self.level] = level_key
                yield sub_keys

    def items(self):
        return self.data.items()

    def get_level(self, key):
        return self.data[key]

    def get_random_combo(self):
        random_key = random.choice(list(self.data.keys()))
        random_value = self.data[random_key]
        if isinstance(random_value, MultiDict):
            combo = self.data[random_key].get_random_combo()
            combo[self.level] = random_key
            return combo
        else:
            return {self.level: random_key}

    def filter_level(self, level_name, keys):
        if self.level == level_name:
            self.data = {k: v for (k, v) in self.data.items() if k in keys}
        elif len(self.level_order) > 1:
            for sub_dict in self.data.values():
                sub_dict.filter_level(level_name, keys)
        else:
            raise Exception('Level "{:s}" not in MultiDict')
