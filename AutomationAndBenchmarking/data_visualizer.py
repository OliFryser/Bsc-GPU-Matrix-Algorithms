import matplotlib.pyplot as pyplot
from csv_data_structure import CSVDataStructure

def visualize_csv(csv_data: CSVDataStructure):
    input_sizes = csv_data.get_input_sizes()
    algorithms = csv_data.get_algorithms()
    for algorithm in algorithms:
        run_times = sorted(csv_data.get_run_times(algorithm))
        print(run_times)
        pyplot.plot(input_sizes, run_times, label=algorithm)
    
    pyplot.title('Run time comparison')
    pyplot.xlabel('Input size')
    pyplot.ylabel('Run time')
    pyplot.xscale('log')
    pyplot.legend()
    pyplot.savefig("Diagrams/output_plot.png")
