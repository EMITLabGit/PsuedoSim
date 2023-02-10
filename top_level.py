import pseudo_analytical_sim
import pprint


def set_NN(simulator):
    input_rows  = NN_layer["Input Height"]
    input_cols  = NN_layer["Input Width"]
    filter_rows = NN_layer["Filter Height"]
    filter_cols = NN_layer["Filter Width"]
    channels    = NN_layer["Channels"]
    num_filter  = NN_layer["Num Filter"]
    stride      = NN_layer["Strides"]



    all_layers = []
    layer_dict = {"Input Height" : input_rows, "Input Width" : input_cols, "Filter Height" : filter_rows, "Filter Width": filter_cols, \
                    "Channels" : channels, "Num Filter" : num_filter, "Strides" : stride}
    all_layers.append(layer_dict)

def main():
     simulator = pseudo_analytical_sim.hardware_state()
     print("\nSimulating performance of test NN model")
     simulator.set_NN(sim_params_analytical.all_layers)
     organize_hardware()


if __name__ == "__main__":
    print("\n"*5) 
    main()
    print("\n"*5) 

