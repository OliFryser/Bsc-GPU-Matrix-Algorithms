class AlgorithmData:
        name: str
        dimension_to_mean_run_time: dict[int, float]
        dimension_to_standard_deviation: dict[int, float]
        
        def __init__(self, name):
            self.dimension_to_mean_run_time = {}
            self.dimension_to_standard_deviation = {}
            self.name = name

class CSVDataStructure:
    # algorithm_name1: [(dimension, mean, stadard_deviation), (dimension, mean, stadard_deviation)]
    # algorithm_name2: [(dimension, mean, stadard_deviation), (dimension, mean, stadard_deviation)]
    # ...
    data: dict[str, AlgorithmData]
    dimensions: list[int]

    def __init__(self, csv_paths: list[str]) -> None:
        self.data = {}
        self.dimensions = []

        for csv in csv_paths:
            with open(csv, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    self.__add_line(line)
            print(len(self.data.keys()))

    def __add_line(self, line: str):
        if not line: return
        algorithm_name, dimension, mean, standard_deviation, iterations = [value.strip() for value in line.rstrip('\n').split(',')]
        if algorithm_name == "Algorithm": return
        dimension = int(dimension)
        mean = float(mean)
        standard_deviation = float(standard_deviation)
        iterations = int(iterations) # not currently in use

        if algorithm_name not in self.data.keys():
            self.data[algorithm_name] = AlgorithmData(algorithm_name)
        
        self.data[algorithm_name].dimension_to_mean_run_time[dimension] = mean
        self.data[algorithm_name].dimension_to_standard_deviation[dimension] = standard_deviation
        
        if dimension not in self.dimensions:
            self.dimensions.append(dimension)