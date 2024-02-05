class CSVDataStructure:
    __data_points: dict = {}

    def __init__(self, csv_path: str) -> None:
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.__add_line(line)

    def __add_line(self, line: str):
        if len(line) == 0: return
        algorithm_name, input_size, run_time = [value.strip() for value in line.rstrip('\n').split(',')]
        data_point = (input_size, run_time)
        if algorithm_name in self.__data_points:
            self.__data_points[algorithm_name].append(data_point)
        else: 
            self.__data_points[algorithm_name] = [data_point]

    def get_data_points(self, algorithm_name: str):
        return self.__data_points[algorithm_name]