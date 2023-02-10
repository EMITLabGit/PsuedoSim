import pseudo_analytical_sim
import pprint


def main():
     simulator = pseudo_analytical_sim.hardware_state()
     print("\nSimulating performance of NN model \"", sim_params_analytical.NN_file_name, "\"", "\n", sep='')
     simulator.set_NN(sim_params_analytical.all_layers)
     organize_hardware()


if __name__ == "__main__":
    print("\n"*5) 
    main()
    print("\n"*5) 

