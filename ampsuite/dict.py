import copy
from collections.abc import MutableMapping, MutableSequence
import json

def flatten_dict(data, simple=1):
    """
    Flatten a nested dictionary or list of dictionaries into a list of flat dictionaries.

    :param data: Input nested dictionary or list of dictionaries
    :param simple: If 1, treat input as a single dictionary; if not 1, treat as a list of dictionaries or a dictionary of dictionaries
    :return: List of flattened dictionaries
    """
    out = []
    def flatten(x, name='',subgroup=None):
        """
        Recursive helper function to flatten nested structures.

        :param x: Current object to flatten
        :param name: Current key name (with parent keys)
        :param subgroup: List of dictionaries being built
        :return: List of flattened dictionaries
        """
        if subgroup is None:
            subgroup = [{}]
        if isinstance(x, MutableMapping):
            for a in x:
                subgroup = flatten(x[a], ''.join([name, str(a), "_"]), subgroup)
        elif isinstance(x, MutableSequence):
            new_dicts = []
            for a in x:
                for dic in subgroup:
                    newdict = flatten(a, name, copy.deepcopy([dic]))
                    for i in newdict:
                        new_dicts.append(i)
            subgroup = new_dicts   
        else:
            for dic in subgroup:
                dic[name[:-1]] = x
        return subgroup

    if simple == 1:
        formatted_data = flatten(data)
        for i in formatted_data:
            out.append(i)
    else:
        for record in data:
            if isinstance(data, MutableSequence):
                formatted_data = flatten(record)
            elif isinstance(data, MutableMapping):
                formatted_data = flatten(data[record])
                for i in formatted_data:
                    i["_ID_VAR_"] = record

            for i in formatted_data:
                out.append(i)
    return out

def none_to_nan(input_dict):
    """
    Replace None values in a dictionary with numpy.nan.

    :param input_dict: Input dictionary
    :return: Dictionary with None values replaced by numpy.nan
    """
    for key in input_dict:
        if input_dict[key] is None:
            input_dict[key] = np.nan
    return input_dict

def readjson(jsonfile):
    """
    Read a JSON file and return its contents as a Python dictionary.

    :param jsonfile: Path to the JSON file
    :return: Dictionary containing the JSON data
    """
    with open(jsonfile,'r') as f:
        dictionary = json.load(f)
    return dictionary

# def converter(instr):
#     '''
#     PANDAS reads arrays in csv as strings heres a quick fic
#     EXAMPLE:
#     original_pdfs['bin_min_max_x'] = original_pdfs.bin_min_max_x.apply(lambda x: converter(x))
#     dhat_x = original_pdfs.dhat_x[idx_ex]
#     dhat_x = dhat_x.values[0:][0] (have to access this weird way....)
#     '''
#     return np.fromstring(instr[1:-1],sep=' ')

# def multindex_iloc(df, index):
#     label = df.index.levels[0][index]
#     return df.iloc[df.index.get_loc(label)]