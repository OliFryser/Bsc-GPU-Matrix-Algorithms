import matplotlib.pyplot as pyplot
from csv_data_structure import CSVDataStructure

def visualize_csv(csv_data: CSVDataStructure, diagram_save_path: str, algorithms_to_compare: list[str]):
    dimensions = csv_data.dimensions
    
    for algorithm in algorithms_to_compare:
        if algorithm not in csv_data.data.keys(): continue
        data = csv_data.data[algorithm]
        run_time_means: list[float] = []
        standard_deviations: list[float] = []
        for dimension in dimensions:
            if dimension not in data.dimension_to_mean_run_time.keys(): continue
            run_time_means.append(data.dimension_to_mean_run_time[dimension])
            standard_deviations.append(data.dimension_to_standard_deviation[dimension])

        pyplot.plot(dimensions, run_time_means, label=algorithm, marker="^")
        #pyplot.errorbar(dimensions, run_time_means, standard_deviations, linestyle='None', marker='^')
    
    pyplot.title('Run time comparison')
    pyplot.xlabel('Dimension')
    pyplot.ylabel('Mean Run time')
    pyplot.xscale('log', base=2)
    pyplot.yscale('log', base=2)
    pyplot.legend()
    pyplot.savefig(diagram_save_path)
