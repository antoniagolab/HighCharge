import pandas
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

def edit_asfinag_file(filename):

    asfinag_data = pd.read_excel(filename+'.xls', header=2)

    extract_asfinag = asfinag_data[asfinag_data['Unnamed: 0'] == 'S01']
    destination_a = extract_asfinag['Unnamed: 5'].to_list()[0]
    destination_b = extract_asfinag['Unnamed: 5'].to_list()[-7]

    list_ind = asfinag_data['Unnamed: 5'].to_list()

    ind_a = list_ind.index(destination_a)
    ind_b = list_ind.index(destination_b)

    n = len(extract_asfinag)

    asfinag_data.at[ind_a:ind_b, 'Unnamed: 0'] = 'S1A'
    asfinag_data.at[ind_b:(ind_a+n-1), 'Unnamed: 0'] = 'S1B'

    asfinag_data.to_csv(filename+"_edited.csv", index=False)
    return asfinag_data


def create_file_with_maximum_util(folder_file):
    """
    from a folder with multiple .xls-files, this function creates a file with maximum values for each traffic counter
    based on all .xls-files (ASFINAG format)
    :param folder_file: String
    :return: pandas.DataFrame
    """
    # collect all .xls files as dataframes in a list
    # create one reference dataframe where max values are collected
    # for each row of these -> extract from each in list -> get max (iterative -- replace max_val if val > curr_max_val)
    #
    files = [f for f in listdir(folder_file) if
             isfile(join(folder_file, f)) and (f[-4:] == '.xls' and not f[:4] == 'Jahr')]

    processed_files = [edit_asfinag_file(folder_file + '/' + f[:len(f) - 4]) for f in files]

    # finding largest file and making it to reference file
    ind_max_len = np.argmax([len(f) for f in processed_files])
    ref_file = processed_files[ind_max_len]
    old_ref_file = ref_file.copy()
    # unique keys: 'Unnamed: 0', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 6'
    highways = ref_file['Unnamed: 0'].to_list()
    locations = ref_file['Unnamed: 2'].to_list()
    traffic_counter_ids = ref_file['Unnamed: 3'].to_list()
    direction = ref_file['Unnamed: 4'].to_list()
    car_type = ref_file['Unnamed: 6'].to_list()
    indices = ref_file.index.to_list()

    for ij in range(0, len(ref_file)):
        hw = highways[ij]
        lc = locations[ij]
        tc = traffic_counter_ids[ij]
        dir = direction[ij]
        ct = car_type[ij]
        ind = indices[ij]

        current_max_val = -1
        for file in processed_files:
            curr_extract_f = file[
                (file['Unnamed: 0'] == hw) & (file['Unnamed: 2'] == lc) & (file['Unnamed: 3'] == tc) & (
                            file['Unnamed: 4'] == dir) & (file['Unnamed: 6'] == ct)]
            if len(curr_extract_f) > 0:
                if curr_extract_f['Kfz/24h'].to_list()[0] > current_max_val:
                    current_max_val = curr_extract_f['Kfz/24h'].to_list()[0]

        ref_file.at[ind, 'Kfz/24h'] = current_max_val

    file_with_max_vals = ref_file.copy()
    return file_with_max_vals





