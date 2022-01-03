import pandas as pd
import datetime
import matplotlib.pyplot as plt


def calculate_dist_max(av_driving_range):
    """
    function calculating the parameter dist_max based on the average driving range of electric vehicles in the car fleet
    :param av_driving_range: (km) average driving range
    :return:
    """
    return av_driving_range * (2/3) * 1000


def create_frequency_df(df):
    """

    :param df:
    :return:
    """
    column_factor = 'weg_hochrechnungsfaktor_woche'
    column_starttime = 'weg_startzeit'
    df[column_starttime] = pd.to_datetime(df[column_starttime], format='%H:%M:%S').dt.time
    timedelta = datetime.timedelta(minutes=15)
    current_time = datetime.time(hour=0)
    frequency_df = pd.DataFrame()
    for ij in range(0, 96):
        # collect frequencies
        new_datetime = datetime.datetime.combine(datetime.date.today(), current_time) + timedelta
        next_time = new_datetime.time()
        extract_df = df[(df[column_starttime] >= current_time) & (df[column_starttime] < next_time)]
        frequency_df = frequency_df.append({'time': current_time, 'frequency': extract_df[column_factor].sum()},
                                       ignore_index=True)
        current_time = next_time
    frequency_df = frequency_df.set_index('time')
    return frequency_df


def calculate_max_util_params(dist_max):
    """
    function calculating parameters mu and gamma_h based on mobility data acquired during a study
    In course of this, two Distributions of starting times during a day are created. These are referred to as H(t) and
    C(t). H(t) represents the frequency of starting car trips passing a highway or motorway. To filter for these trips
    similar filters as in Jochem et al. (2016) are used. C(t) indicates this distribution for cars needing to be charged
    along the trip which is defined based on the parameter dist_max. H(t) basically reflects the traffic count on a
    highway for a given point in time.
    :param dist_max: (m)
    :return: mu, gamma_h
    """
    way_data = pd.read_csv('mobility_data\ÖU_2013-2014_Datensatz_CSV_Version/ÖU_2013_2014_Wegedatensatz.csv')

    # filter for ways made primarily by car with person as driver
    column_car_driver = 'weg_vm_haupt'
    car_drivers_data = way_data[way_data[column_car_driver] == 301]

    column_trip_time = 'weg_dauer'
    column_trip_distance = 'weg_laenge'
    column_starttime = 'weg_startzeit'
    # extracting all trips H(t)
    h_t = car_drivers_data[((car_drivers_data[column_trip_time] >= 45) & (car_drivers_data[column_trip_distance] >= 25)) |
                           (car_drivers_data[column_trip_distance] >= 50)]

    # extracting all trips C(t)
    c_t = car_drivers_data[car_drivers_data[column_trip_distance] >= (dist_max/1000)]


    h_t = h_t[~(h_t[column_starttime] == ' ')]
    c_t = c_t[~(c_t[column_starttime] == ' ')]

    # goal: for each time step of the day: maximum
    h_t_counts = create_frequency_df(h_t)
    c_t_counts = create_frequency_df(c_t)


    # plot
    fig, ax = plt.subplots()
    h_t_counts.plot(ax=ax)
    c_t_counts.plot(ax=ax)

    # find peak hour by sliding a window over time

    # need to slide in timedelta steps
    timedelta = datetime.timedelta(minutes=15)
    current_max = 0
    current_max_starttime = datetime.time(hour=0)

    current_time = datetime.time(hour=0)    # initial lower bound

    smoothed_values_c = []
    smoothed_values_h = []

    current_max_h = 0
    for ij in range(0, 96):
        # upper bound
        top_datetime = datetime.datetime.combine(datetime.date.today(), current_time) + datetime.timedelta(hours=1)
        top_time = top_datetime.time()
        extract_df_h_t = h_t_counts[(h_t_counts.index >= current_time) & (h_t_counts.index < top_time)]
        extract_df_c_t = c_t_counts[
            (c_t_counts.index >= current_time) & (c_t_counts.index < top_time)]
        # print(extract_df_c_t.frequency.sum())
        smoothed_values_c.append(extract_df_c_t.frequency.sum())
        smoothed_values_h.append(extract_df_h_t.frequency.sum())
        if extract_df_c_t.frequency.sum() > current_max:
            current_max_starttime = current_time
            current_max = extract_df_c_t.frequency.sum()
            current_max_h = extract_df_h_t.frequency.sum()

        # getting next lower bound
        new_datetime = datetime.datetime.combine(datetime.date.today(), current_time) + timedelta
        next_time = new_datetime.time()
        current_time = next_time

    # downscaling to the peak hour
    #   - assuming: sum over all trips in H(t) = 100% traffic count -> how much percent is H(t) during peak hour?
    gamma_h = current_max_h/h_t_counts.frequency.sum()
    mu = current_max/current_max_h

    return mu, gamma_h
