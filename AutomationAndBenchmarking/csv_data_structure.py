class CSVDataStructure:
    __run_times: dict = {}
    __input_sizes: list = []

    def __init__(self, csv_path: str) -> None:
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.__add_line(line)

    def __add_line(self, line: str):
        if not line: return
        algorithm_name, input_size, run_time = [value.strip() for value in line.rstrip('\n').split(',')]
        input_size = float(input_size)
        run_time = float(run_time)
        
        if algorithm_name in self.get_algorithms() and input_size in self.get_input_sizes():
            return

        if input_size not in self.__input_sizes:
            self.__input_sizes.append(input_size)

        if algorithm_name in self.__run_times:
            self.__run_times[algorithm_name].append(run_time)
        else: 
            self.__run_times[algorithm_name] = [run_time]

    def get_run_times(self, algorithm_name: str):
        return self.__run_times[algorithm_name]
    
    def get_algorithms(self):
        return self.__run_times.keys()
    
    def get_input_sizes(self):
        return sorted(self.__input_sizes)