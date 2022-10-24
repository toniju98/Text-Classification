import pandas as pd
import os
import json


def read_data(data1, data2):
    """loads data from files in a dataframe

    :param path: path, where the files are
    :return: data, pd.Dataframe with the whole data in it
    """
    data = pd.concat([data1, data2], ignore_index=True)
    data[['url', 'topic', 'headline', 'summary', 'date', 'text']] = data[
        ['url', 'topic', 'headline', 'summary', 'date', 'text']].astype('str')
    return data


def reading_data_day(path):
    """reading spiegel data

    :param path: path, where the files are
    :return: spiegel data
    """
    with open(path, encoding="utf8") as data_file:
        d = json.load(data_file)
    spiegel_data = pd.DataFrame(d['articles'])
    return spiegel_data


def other_data(path):
    """reading other data from other api

        :param path: path, where the files are
        :return: data
        """
    with open(path, encoding="utf8") as data_file:
        d = json.load(data_file)
    data = pd.DataFrame(d['articles'])
    source_list = []
    for i in range(len(d['articles'])):
        source_list.append(d['articles'][i]['source']['name'])
    data['source'] = source_list
    return data


def reading_data_week(path):
    """reading sz data

        :param path: path, where the files are
        :return: sz data
        """
    sz_data = pd.DataFrame()
    with open(path, encoding="utf8") as data_file:
        d = json.load(data_file)
    for i in range(len(d['articles'])):
        sz_onepart = pd.DataFrame(d['articles'][i])
        sz_data = pd.concat([sz_data, sz_onepart], ignore_index=True)
    return sz_data


def get_alljsonfrompath(path):
    """gets list of paths from all jsons in a directory

        :param path: path of the directory
        :return: list of all json paths
        """
    all_json_paths = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".json")):
                full_path = os.path.join(root, name)
                all_json_paths.append(full_path)
    return all_json_paths
