import csv
from typing import Iterator
from datetime import datetime
import time
import math

from crossdb.postgresql.psql_trip import PsqlTrip
from crosstrips.models.trips.trip import Trip


class DataService:
    def __init__(self, psql_trip: PsqlTrip, trip_service, database_repository, ceph_travel_repository):
        self.psql_trip = psql_trip
        self.trip_service = trip_service
        self.database = database_repository
        self.ceph_travel = ceph_travel_repository

    def get_all_data(self, year_int):
        aggressive_count = 0
        total_count = 0
        skipped_count = 0
        filename = str(year_int) + "_" + time.strftime("%Y%m%d-%H%M%S") + '_prod_driving_behavior_research.csv'
        wtr = csv.writer(open(filename, 'w'), delimiter=',',
                         lineterminator='\n')
        # err_wtr = csv.writer(open(time.strftime("%Y%m%d-%H%M%S") + '_prod_driving_behavior_research_errors.csv', 'w'),
        #                      delimiter=',', lineterminator='\n')
        wtr.writerow(
            ["trip_id", "aggressive_acceleration", "speeding", "speeding_binary", "aggressive_behavior_combined", 'road_type', 'travelled_distance', 'speed_limit',
             "rush_hour",
             "vehicle_id", 'car_category', 'power', 'acceleration', 'composition', 'fuel_type', 'model_year',
             'unladen_mass',
             'euro_classification', 'fuel_consumption_combined', 'energy_label', 'top_speed', 'weather_comfort',
             'dewPoint',
             'skyInfo', 'daylight', 'distance', 'humidity', "windSpeed", 'visibility', 'temperature'])

        #define the fine height for urban and non-urban areas
        fine_in = [0, 0, 0, 29, 36, 43, 50, 57, 65, 74, 100, 108, 120, 129, 139, 149, 160, 173, 184, 197, 211, 224, 239,
                   251, 267, 282, 300, 314, 330, 346]
        fine_out = [0, 0, 0, 25, 32, 39, 46, 52, 61, 69, 94, 104, 114, 123, 133, 142, 152, 162, 176, 188, 200, 212, 224,
                    239, 252, 267, 280, 295, 313, 329]

        # update all dicts
        car_category_dict = {None: 0, 'MOTORCYCLE': 1, 'TRUCK_MIDDLE_WEIGHT': 2, 'LCV': 3, 'CAR': 4, 'TRUCK_HEAVY': 5}
        car_composition_dict = {None: 0, 'Hatchback': 2, 'Verv Voertuigen': 3, 'Sedan': 4, 'Cabriolet': 5,
                                'Coupe': 6, 'Cabrio': 7, 'Stationwagen': 8, 'MPV': 9, 'Gesloten Wagen': 10, 'Gecond. Voertuig': 11,
                                'Trekker': 12, 'Kampeerwagen': 13, 'MVP': 14}
        fuel_type_dict = {None: 0, 'D': 1, 'B': 2, 'Waterstof': 3, 'G': 4, 'Benzine': 5, 'E': 6, '': 7}
        energy_label_dict = {None: 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, '': 8}

        devices = self.psql_trip.get_devices()
        for device in devices:
            vehicle_info = self.ceph_travel.get_vehicle_info_from_cs_vehicle(device)
            if vehicle_info:
                vehicle_info = vehicle_info[0]
                if vehicle_info[2] not in car_composition_dict.keys() or vehicle_info[1] not in car_category_dict.keys() or vehicle_info[5] not in fuel_type_dict or vehicle_info[10] not in energy_label_dict:
                    # if unknown values occur skip this one (not in the first 800000 rows)
                    continue
                if vehicle_info[1] not in ['CAR', 'LCV']:
                    # for now only allow Cars and LCV
                    continue
                # make sure the vehicle info is complete
                vehicle_info_used = [vehicle_info[0], car_category_dict[vehicle_info[1]],
                                          car_composition_dict[vehicle_info[2]],
                                          vehicle_info[3], vehicle_info[4], fuel_type_dict[vehicle_info[5]],
                                          vehicle_info[6], vehicle_info[7],
                                          vehicle_info[8], vehicle_info[9], energy_label_dict[vehicle_info[10]],
                                          vehicle_info[11]]
                if None in vehicle_info_used or '' in vehicle_info:
                    # empty value found so skip this one
                    continue
                all_trips = []
                if year_int == 2017:
                    all_trips = list(self.get_recent_trips_1(device))
                elif year_int == 2018:
                    all_trips = list(self.get_recent_trips_2(device))
                elif year_int == 2019:
                    all_trips = list(self.get_recent_trips_3(device))
                elif year_int == 2020:
                    all_trips = list(self.get_recent_trips_4(device))
                for trip in all_trips:
                    if "weather" not in trip.df or type(trip.df['weather'].values[0]) is not dict:
                        break
                    # get weather from first weather row
                    weather = trip.df['weather'].values[0]
                    if type(weather) is not dict:
                        if math.isnan(weather):
                            # sometimes the first weather record is not filled the last one can be used if filled
                            weather_size = len(trip.df['weather'].values) - 1
                            if type(trip.df['weather'].values[weather_size]) is not dict:
                                break
                            else:
                                weather = trip.df['weather'].values[weather_size]

                    point_count = 0
                    for i in range(len(trip.df['speed'].values)):
                        aggressive_combined = 0
                        aggressive_acceleration = 0
                        # trip.df['aggressive_combined'] = (trip.df['acceleration'].between(-2.7, 2.7) == False)
                        if trip.df['acceleration'].values[i] > 2.7 or trip.df['acceleration'].values[i] < -2.7:
                            aggressive_acceleration = 1
                            aggressive_combined = 1
                        speed_limit = trip.df['speed_limit'].values[i]
                        above_speed_limit = trip.df['speed'].values[i] - speed_limit
                        speeding = 0
                        # for the first 3 k/m overrun no fine is given, and a correction of 3 is calculated
                        if above_speed_limit > 6:
                            # The fines in the netherlands are known till overrun by 30 km/h
                            above_speed_limit = above_speed_limit if above_speed_limit < 30 else 30
                            speeding = fine_in[above_speed_limit - 4] if speed_limit == 30 else fine_out[
                                above_speed_limit - 4]
                            aggressive_combined = 1

                        # update weather if new data is available
                        if type(trip.df['weather'].values[i]) is dict:
                            weather = trip.df['weather'].values[i]

                        rush_hour = ""
                        # Check if startdate is valid
                        # Could use trip.df.index.values[i] to get numpy datetime object
                        rush_hour = "1" if trip.startdate.hour in (7, 8, 16, 17) else "0"

                        daylight = 0
                        #unused take care of equal sample size
                        # aggressive_count += aggressive_combined
                        # if total_count/2 > (aggressive_count+10) and aggressive_combined == 0:
                        #     #Log how many rows are skipped at this check
                        #     skipped_count +=1
                        #     print('Amount of skipped points is: ' + str(skipped_count))
                        #     continue
                        total_count += 1

                        try:
                            wtr.writerow([trip.id, aggressive_acceleration, speeding, str(1) if speeding > 0 else 0,
                                          aggressive_combined,
                                          trip.df['road_type'].values[i], trip.df['travelled_distance'].values[i],
                                          trip.df['speed_limit'].values[i],
                                          rush_hour, vehicle_info[0], car_category_dict[vehicle_info[1]],
                                          car_composition_dict[vehicle_info[2]],
                                          vehicle_info[3], vehicle_info[4], fuel_type_dict[vehicle_info[5]],
                                          vehicle_info[6], vehicle_info[7],
                                          vehicle_info[8], vehicle_info[9], energy_label_dict[vehicle_info[10]],
                                          vehicle_info[11],
                                          weather['comfort'], weather['dewPoint'],
                                          weather['skyInfo'], daylight, weather['distance'], weather['humidity'],
                                          weather['windSpeed'], weather['visibility'], weather['temperature']])
                        except Exception as error:
                           print([device, trip.id, point_count, error])
                        point_count += 1

        return filename





    def get_recent_trips_1(self, device_id: str) -> Iterator[Trip]:
        yield from self.trip_service.find_trips_by_device_id_and_dates(device_id, '2017-01-01 00:00:00',
                                                                           '2018-01-01 00:00:00')

    def get_recent_trips_2(self, device_id: str) -> Iterator[Trip]:
        yield from self.trip_service.find_trips_by_device_id_and_dates(device_id, '2018-01-01 00:00:00',
                                                                   '2019-01-01 00:00:00')
    def get_recent_trips_3(self, device_id: str) -> Iterator[Trip]:
        yield from self.trip_service.find_trips_by_device_id_and_dates(device_id, '2019-01-01 00:00:00', '2020-01-01 00:00:00')

    def get_recent_trips_4(self, device_id: str) -> Iterator[Trip]:
        yield from self.trip_service.find_trips_by_device_id_and_dates(device_id, '2020-01-01 00:00:00',
                                                                               '2021-01-01 00:00:00')
