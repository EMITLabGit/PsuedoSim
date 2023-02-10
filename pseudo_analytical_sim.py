import math
import time
#from regex import B
import SRAM_model
import pandas as pd
num_conv_in_input_delete = 0
#import sim_params_analytical
import specs_info


print_string = 0
input_sram_time = 0
filter_sram_time = 0

class hardware_state():
     def __init__(self):
          x = 1

     def set_hardware(self, hardware_state_info):
          self.array_rows = hardware_state_info.loc["Systolic Array Rows"].item()
          self.array_cols = hardware_state_info.loc["Systolic Array Cols"].item()
          self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"].item()
          self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"].item()
          self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"].item()
          self.accumulator_elements = hardware_state_info.loc["Accumulator Elements"].item()
          self.batch_size = hardware_state_info.loc["Batch Size"].item()

          '''
          print("\n---------------------------------")
          print("Now setting hardware to:")
          print("Array Size:       ", self.array_rows, "x", self.array_cols)
          print("SRAM Input Size:  ", self.SRAM_input_size)
          print("SRAM Filter Size: ", self.SRAM_filter_size)
          print("SRAM Output Size: ", self.SRAM_output_size)
          print("Accumulator Elements per Col: ", self.accumulator_elements)
          print("Batch Size: ", self.batch_size)
          print("---------------------------------")
          print()
          '''
          self.input_SRAM  = SRAM_model.SRAM_model(self.SRAM_input_size, "input")
          self.filter_SRAM = SRAM_model.SRAM_model(self.SRAM_filter_size, "filter")
     
     def set_NN(self, NN_layers_all):
          self.NN_layers_all = NN_layers_all
          self.num_NN_layers = len(NN_layers_all)

     def set_results_vars(self):
          self.num_programming_practice = [0] * self.num_NN_layers
          self.num_programming_theory   = [0] * self.num_NN_layers
          self.num_compute_cycles_practice = [0] * self.num_NN_layers
          self.num_compute_cycles_theory = [0] * self.num_NN_layers

          self.SRAM_input_reads = [0] * self.num_NN_layers
          self.SRAM_filter_reads = [0] * self.num_NN_layers
          self.SRAM_output_writes = [0] * self.num_NN_layers
          self.DRAM_input_reads = [0] * self.num_NN_layers
          self.DRAM_filter_reads = [0] * self.num_NN_layers
          self.DRAM_output_writes = [0] * self.num_NN_layers
          self.SRAM_DRAM_input_misses = [0] * self.num_NN_layers
          self.SRAM_DRAM_filter_misses = [0] * self.num_NN_layers
          self.SRAM_input_accesses = [0] * self.num_NN_layers
          self.SRAM_filter_accesses = [0] * self.num_NN_layers
          self.accumulator_dumps = [0] * self.num_NN_layers

     def run_all_layers(self):
          self.set_results_vars()
          start_time = time.time()
          global print_string
          print_string_base = "Now simulating layer "
          print_string = print_string_base
          for index, layer in enumerate(self.NN_layers_all):
               print_string += str(index)
               print_string = print_string_base + str(index)
               print(print_string, end="\r", flush=True)
               print_string += ", "
               self.current_layer = index
               global input_sram_time
               global filter_sram_time
               input_sram_time = 0
               filter_sram_time = 0
               status = self.single_layer_set_params(layer)
               if (status == -1):
                    return
          #print()
          end_time = time.time()
          print("Done with single simulation, it took", round((end_time - start_time) / 60, 2), " minutes                          ")
          self.access_SRAM_data()
          self.calculate_NN_totals()
          #self.print_layer_results()
          #self.print_NN_results()
          self.save_all_layers_csv()
          return(self.return_specs())

     def single_layer_set_params(self, NN_layer):
          input_rows  = NN_layer["Input Height"]
          input_cols  = NN_layer["Input Width"]
          filter_rows = NN_layer["Filter Height"]
          filter_cols = NN_layer["Filter Width"]
          channels    = NN_layer["Channels"]
          num_filter  = NN_layer["Num Filter"]
          stride      = NN_layer["Strides"]

          filter_size = filter_rows * filter_cols * channels
          single_input_size = input_cols * input_rows * channels

          overall_col_fold = math.ceil(num_filter / self.array_cols)  
          overall_row_fold = math.ceil(filter_size / self.array_rows)
          overall_batch_fold = self.batch_size

          conv_rows = math.ceil((input_rows - filter_rows) / stride) + 1
          conv_cols = math.ceil((input_cols - filter_cols) / stride) + 1
          #conv_rows = math.ceil(input_rows / stride)
          #conv_cols = math.ceil(input_cols / stride)

          num_conv_in_input = conv_rows * conv_cols 
          global num_conv_in_input_delete 
          num_conv_in_input_delete = num_conv_in_input
          batch_col_fold_product_allowed = self.accumulator_elements / num_conv_in_input # what can fit onto accumulator
          if (batch_col_fold_product_allowed < 1):
               print("CANNOT DO SIMULATION B/C ACCUMULATOR NOT BIG ENOUGH FOR ONE BATCH WITH ONE COL FOLD")
               print("Current Layer: ", self.current_layer)
               return(-1)


          minor_col_fold = 1
          minor_batch_fold = math.floor(batch_col_fold_product_allowed / minor_col_fold)
          if minor_batch_fold > overall_batch_fold:
               minor_batch_fold = overall_batch_fold
               minor_col_fold = math.floor(batch_col_fold_product_allowed / minor_batch_fold)
               if minor_col_fold > overall_col_fold:
                    minor_col_fold = overall_col_fold
          '''
          minor_batch_fold = 1
          minor_col_fold = math.floor(batch_col_fold_product_allowed / minor_batch_fold)
          if minor_col_fold > overall_col_fold:
               minor_col_fold = overall_col_fold
          '''
          minor_row_fold = 1

          major_col_fold = math.ceil(overall_col_fold / minor_col_fold)
          major_row_fold = math.ceil(overall_row_fold / minor_row_fold)
          major_batch_fold = math.ceil(overall_batch_fold / minor_batch_fold)

          self.accumulator_dumps[self.current_layer] = major_batch_fold * major_col_fold
          self.num_compute_cycles_theory[self.current_layer] = self.batch_size * overall_col_fold * overall_row_fold * num_conv_in_input
          self.SRAM_input_reads[self.current_layer] = num_conv_in_input * self.batch_size * filter_size * overall_col_fold
          self.SRAM_filter_reads[self.current_layer] = filter_size * num_filter * major_batch_fold
          self.SRAM_output_writes[self.current_layer] = num_conv_in_input * self.batch_size * num_filter
          if self.current_layer == (self.num_NN_layers - 1):
               self.DRAM_output_writes[self.current_layer] = self.SRAM_output_writes[self.current_layer]
          else:
               self.DRAM_output_writes[self.current_layer] = max(self.SRAM_output_writes[self.current_layer] - self.SRAM_output_size, 0)

          if self.current_layer == 0:
               SRAM_input_output_crossover_data = 0
          else:
               SRAM_input_output_crossover_data = min(self.SRAM_output_size, self.SRAM_output_writes[self.current_layer - 1])

          new_mem = self.input_SRAM.new_layer(single_input_size, self.batch_size, SRAM_input_output_crossover_data)
          if new_mem == -1:
               return -1
          new_mem = self.filter_SRAM.new_layer(self.array_cols * self.array_rows, overall_row_fold * overall_col_fold, 0)
          if new_mem == -1:
               return -1
          self.run_single_layer(major_col_fold, minor_col_fold, overall_col_fold, \
               major_batch_fold, minor_batch_fold, overall_batch_fold,\
                     major_row_fold, minor_row_fold, overall_row_fold)
          if (self.num_programming_theory != self.num_programming_practice):
               print("Theoretical Programming Count NOT EQUAL to Practical Programming Count")
               return(-1)
          self.input_SRAM.conclude_layer()
          self.filter_SRAM.conclude_layer()



     def run_single_layer(self, major_col_fold, minor_col_fold, overall_col_fold,\
           major_batch_fold, minor_batch_fold, overall_batch_fold,\
               major_row_fold, minor_row_fold, overall_row_fold):     
          self.num_programming_practice[self.current_layer] = 0
          self.num_programming_theory[self.current_layer] = overall_col_fold * overall_row_fold * major_batch_fold
          if (overall_row_fold == 1) :
               self.num_programming_theory[self.current_layer] = overall_col_fold

          global print_string
          old_col_group = -1
          old_row_group = -1
          current_col_group = -1
          current_row_group = -1
          num_loop_iterations = overall_col_fold * overall_row_fold * overall_batch_fold
          current_loop_iteration = 0
          target_inc = 10
          target = target_inc
          for major_col_group in range(major_col_fold):
               for major_batch_group in range(major_batch_fold):
                    for minor_col_group in range(minor_col_fold):
                         for major_row_group in range(major_row_fold):
                              for minor_row_group in range(minor_row_fold):
                                   for minor_batch_group in range(minor_batch_fold):
                                        current_loop_iteration += 1
                                        percent_done = 100 * current_loop_iteration / num_loop_iterations
                                        if percent_done >= target:
                                             addon = "percent done: "+ str(round(percent_done)) + "  "
                                             print_string += addon
                                             print(print_string, end="\r", flush=True)
                                             print_string = print_string[0:len(print_string) - len(addon)]
                                             target += target_inc

                                        old_col_group = current_col_group
                                        old_row_group = current_row_group

                                        current_col_group = major_col_group * minor_col_fold + minor_col_group
                                        current_row_group = major_row_group * minor_row_fold + minor_row_group
                                        current_batch = major_batch_group * minor_batch_fold + minor_batch_group

                                        if current_col_group >= overall_col_fold:
                                             continue # could probably be "break" but just to play it safe do this
                                        if current_row_group >= overall_row_fold:
                                             continue
                                        if current_batch >= overall_batch_fold:
                                             continue

                                        if (old_col_group != current_col_group) or (old_row_group != current_row_group):
                                             self.num_programming_practice[self.current_layer] += 1
                                        self.num_compute_cycles_practice[self.current_layer] += num_conv_in_input_delete 

                                        if (0):
                                             print("")
                                             print("Current Col Group:", current_col_group)
                                             print("Current Row Group:", current_row_group)
                                             print("Current Batch Group:", current_batch)

                                        filter_index = current_col_group * major_row_fold * minor_row_fold + current_row_group
                                        self.manage_SRAM_DRAM_access(current_batch, filter_index)

          #self.SRAM_filter_reads[self.current_layer] = self.num_programming_practice[self.current_layer] * self.
          

     def manage_SRAM_DRAM_access(self, current_batch, filter_index):
          start_time = time.time()
          self.input_SRAM.access_component(current_batch)
          med_time = time.time()
          self.filter_SRAM.access_component(filter_index)
          end_time = time.time()
          global input_sram_time, filter_sram_time
          input_sram_time += med_time - start_time
          filter_sram_time += end_time - med_time

     def access_SRAM_data(self):
          self.DRAM_input_reads  = self.input_SRAM.DRAM_reads
          self.DRAM_filter_reads = self.filter_SRAM.DRAM_reads

          self.SRAM_DRAM_input_misses  = self.input_SRAM.component_misses
          self.SRAM_DRAM_filter_misses = self.filter_SRAM.component_misses
          self.SRAM_input_accesses = self.input_SRAM.component_accesses
          self.SRAM_filter_accesses = self.filter_SRAM.component_accesses

          self.input_SRAM.conclude_NN()
          self.filter_SRAM.conclude_NN()

     def calculate_NN_totals(self):
          self.num_programming_practice_total    = sum(self.num_programming_practice)
          self.num_programming_theory_total      = sum(self.num_programming_theory)
          self.num_compute_cycles_practice_total = sum(self.num_compute_cycles_practice)
          self.num_compute_cycles_theory_total   = sum(self.num_compute_cycles_theory)

          self.SRAM_input_reads_total   = sum(self.SRAM_input_reads)
          self.SRAM_filter_reads_total  = sum(self.SRAM_filter_reads)
          self.SRAM_output_writes_total = sum(self.SRAM_output_writes)
          self.DRAM_input_reads_total   = sum(self.DRAM_input_reads)
          self.DRAM_filter_reads_total  = sum(self.DRAM_filter_reads)
          self.DRAM_output_writes_total = sum(self.DRAM_output_writes)
          self.accumulator_dumps_total  = sum(self.accumulator_dumps)

          self.SRAM_DRAM_input_misses_total  = sum(self.SRAM_DRAM_input_misses)
          self.SRAM_DRAM_filter_misses_total = sum(self.SRAM_DRAM_filter_misses)
          self.SRAM_input_accesses_total = sum(self.SRAM_input_accesses)
          self.SRAM_filter_accesses_total = sum(self.SRAM_filter_accesses)

     def print_NN_results(self):
          print("\n-----------Total Results Across all Layers-----------")
          print("Num Programming Practice: ", self.num_programming_practice_total)
          print("Num Programming Theory:  ", self.num_programming_theory_total)  
          print("Num Compute Cycles Practice: ", self.num_compute_cycles_practice_total)
          print("Num Compute Cycles Theory: ", self.num_compute_cycles_theory_total)

          print("SRAM Input Reads: ", self.SRAM_input_reads_total)
          print("SRAM Filter Reads: ", self.SRAM_filter_reads_total)
          print("SRAM Output Writes: ", self.SRAM_output_writes_total)
          print("DRAM Input Reads: ", self.DRAM_input_reads_total)
          print("DRAM Filter Reads: ", self.DRAM_filter_reads_total)
          print("DRAM Output Writes: ", self.DRAM_output_writes_total)
          print("Accumulator Dumps: ", self.accumulator_dumps_total)

          print("SRAM DRAM Input  Misses: ", self.SRAM_DRAM_input_misses_total)
          print("SRAM DRAM Filter Misses: ", self.SRAM_DRAM_filter_misses_total)
          print("SRAM Input  Accesses : ", self.SRAM_input_accesses_total)
          print("SRAM Filter Accesses : ", self.SRAM_filter_accesses_total)

     def print_layer_results(self):
          for layer_num in range(self.num_NN_layers):
               print("\n----Results for layer", str(layer_num), "----")
               print("Num Programming Practice: ", self.num_programming_practice[layer_num])
               print("Num Programming Theory:  ", self.num_programming_theory[layer_num])  
               print("Num Compute Cycles Practice: ", self.num_compute_cycles_practice[layer_num])
               print("Num Compute Cycles Theory: ", self.num_compute_cycles_theory[layer_num])

               print("SRAM Input Reads: ", self.SRAM_input_reads[layer_num])
               print("SRAM Filter Reads: ", self.SRAM_filter_reads[layer_num])
               print("SRAM Output Writes: ", self.SRAM_output_writes[layer_num])
               print("DRAM Input Reads: ", self.DRAM_input_reads[layer_num])
               print("DRAM Filter Reads: ", self.DRAM_filter_reads[layer_num])
               print("DRAM Output Writes: ", self.DRAM_output_writes[layer_num])
               print("Accumulator Dumps: ", self.accumulator_dumps[layer_num])

               print("SRAM DRAM Input  Misses: ", self.SRAM_DRAM_input_misses[layer_num])
               print("SRAM DRAM Filter Misses: ", self.SRAM_DRAM_filter_misses[layer_num])
               print("SRAM Input  Accesses : ", self.SRAM_input_accesses_total[layer_num])
               print("SRAM Filter Accesses : ", self.SRAM_filter_accesses_total[layer_num])

     '''
     def save_all_layers_csv(self):
          data_together = [self.num_programming_practice, self.num_programming_theory, self.num_compute_cycles_practice, \
               self.num_compute_cycles_theory, self.SRAM_input_reads, self.SRAM_filter_reads, self.SRAM_output_writes, \
                    self.DRAM_input_reads, self.DRAM_filter_reads, self.DRAM_output_writes, self.accumulator_dumps, \
                         self.SRAM_DRAM_input_misses, self.SRAM_DRAM_filter_misses, self.SRAM_input_accesses, self.SRAM_filter_accesses]
          classes = ["Num Programming Practice", "Num Programming Theory", "Num Compute Cycles Practice", \
               "Num Compute Cycles Theory", "SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
                    "DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes", "Accumulator Dumps", \
                         "SRAM DRAM Input  Misses (Batch)", "SRAM DRAM Filter Misses (Batch)", "SRAM Input Accesses", "SRAM Filter Accesses"]

          totals = [self.num_programming_practice_total, self.num_programming_theory_total, self.num_compute_cycles_practice_total, \
               self.num_compute_cycles_theory_total, self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_total, \
                    self.DRAM_input_reads_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_total, self.accumulator_dumps_total, \
                         self.SRAM_DRAM_input_misses_total, self.SRAM_DRAM_filter_misses_total, self.SRAM_input_accesses_total, self.SRAM_filter_accesses_total]
          df = pd.DataFrame(data_together, classes)
          df.insert(df.shape[1], "totals", totals)
          name_add_on = self.hardware_name()
          df.to_csv(sim_params_analytical.detailed_results_folder_complete + name_add_on)
          x = 1
     '''

     def hardware_name(self):
          name = "_BS_" + str(self.batch_size) + "_AS_" + str(self.array_rows) + "_" + str(self.array_cols) +\
                "_SRAM_" + str(self.SRAM_input_size) + "_" + str(self.SRAM_filter_size) + "_" + str(self.SRAM_output_size) + \
                "_ACC_" + str(self.accumulator_elements) + ".csv"
          return(name)

     def return_specs(self):
          totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_total, \
                    self.DRAM_input_reads_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_total, \
                    self.num_programming_theory_total, self.num_compute_cycles_theory_total, self.accumulator_dumps_total]

          return(pd.DataFrame(totals, specs_info.runspecs_names))
          x= 2