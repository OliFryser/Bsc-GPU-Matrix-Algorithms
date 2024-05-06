import matplotlib.pyplot as pyplot
from csv_data_structure import CSVDataStructure
import numpy

def visualize_csv(csv_data: CSVDataStructure, diagram_save_path: str, algorithms_to_compare: list[str]):
    dimensions = csv_data.dimensions

    for algorithm in algorithms_to_compare:
        if algorithm not in csv_data.data.keys(): continue
        data = csv_data.data[algorithm]
        run_time_means: list[float] = []
        for dimension in dimensions:
            if dimension not in data.dimension_to_mean_run_time.keys(): continue
            run_time_means.append(data.dimension_to_mean_run_time[dimension])

        if("cpu" in algorithm):
            b, a = numpy.polyfit(numpy.log2(dimensions), numpy.log2(run_time_means), deg=1)
            algorithm_label = f'{algorithm}: y = {round(b, 2)} x' + ("" if a < 0 else "+") + f'{round(a, 2)}'
            pyplot.scatter(dimensions, run_time_means, label=algorithm_label, marker="^")
            pred_f = a + numpy.multiply(numpy.log2(dimensions), b)
            pyplot.plot(sorted(dimensions), numpy.exp2(pred_f), 'k--')
        else:
            pyplot.plot(dimensions, run_time_means, label=algorithm, marker="^")

    pyplot.xlabel('Matrix Side Length')
    pyplot.ylabel('Mean Run Time (in seconds)')
    pyplot.xscale('log', base=2)
    pyplot.yscale('log', base=2)
    pyplot.legend()
    pyplot.savefig(diagram_save_path)
