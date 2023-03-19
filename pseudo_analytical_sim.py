import math
import time
import SRAM_model
import pandas as pd

print_string = 0
input_sram_time = 0
filter_sram_time = 0

class hardware_state():
	def __init__(self):
		x = 1

	def set_hardware(self, hardware_state_info):
		self.array_rows = hardware_state_info.loc["Systolic Array Rows"].item()
		self.array_cols = hardware_state_info.loc["Systolic Array Cols"].item()
		self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"].item() * 1000 / 2
		self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"].item() * 1000 / 2
		self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"].item() * 1000 / 2
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
		self.output_SRAM = SRAM_model.SRAM_model(self.SRAM_output_size, "output")


	def set_NN(self, NN_layers_all):
		self.NN_layers_all = NN_layers_all
		self.num_NN_layers = len(NN_layers_all)

	def set_results_vars(self):
		self.num_program_compute_instance = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_analog = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_digital = [0] * self.num_NN_layers
		self.num_program_clock_cycles = [0] * self.num_NN_layers

		self.SRAM_input_reads = [0] * self.num_NN_layers
		self.SRAM_filter_reads = [0] * self.num_NN_layers
		self.SRAM_output_writes = [0] * self.num_NN_layers
		self.SRAM_output_reads = [0] * self.num_NN_layers

		self.DRAM_input_reads_analytical = [0] * self.num_NN_layers
		self.DRAM_filter_reads_analytical = [0] * self.num_NN_layers 
		self.DRAM_output_writes_analytical = [0] * self.num_NN_layers
		self.DRAM_output_reads_analytical = [0] * self.num_NN_layers

		self.DRAM_input_reads_SRAM_sharing = [0] * self.num_NN_layers
		self.DRAM_output_writes_SRAM_sharing = [0] * self.num_NN_layers

		#self.DRAM_input_reads_total   = 0
		#self.DRAM_filter_reads_total  = 0
		#self.DRAM_output_writes_total = 0
		#self.DRAM_output_reads_total  = 0

		#self.DRAM_input_reads_SRAM_sharing_total = 0
		#self.DRAM_output_writes_SRAM_sharing_total = 0


	def run_all_layers(self):
		print("\nBeginning analytical modeling simulation")
		self.set_results_vars()
		start_time = time.time()
		global print_string
		print_string_base = "Now simulating layer "
		print_string = print_string_base
		for index, layer in enumerate(self.NN_layers_all):
			print_string += str(index)
			print_string = print_string_base + str(index)
			#print(print_string, end="\r", flush=True)
			print_string += ", "
			self.current_layer = index
			global input_sram_time
			global filter_sram_time
			input_sram_time = 0
			filter_sram_time = 0
			self.SRAM_carryover_data_current_layer = 0; self.SRAM_carryover_data_previous_layer = 0
			status = self.single_layer_set_params(layer)
			if (status == -1):
				return
		end_time = time.time()
		AM_execution_time = round((end_time - start_time) / 60, 2)
		#print("Done with simulation, it took", AM_execution_time, "minutes                          ")
		self.calculate_NN_totals()
		final_time = time.time()
		AM_post_process_time = round((final_time - end_time) / 60, 2)
		#self.print_layer_results()
		#self.print_NN_results()
		#self.save_all_layers_csv()
		AM_results = self.return_specs()
		AM_results.loc[" "] = " "
		AM_results.loc["Simulation Run Time [min]"] = AM_execution_time
		AM_results.loc["Simulation Post Process Time [min]"] = AM_post_process_time
		return(AM_results)
	

	def compute_analytical_expressions(self, num_conv_in_input, col_fold, row_fold, ind_filter_size, num_filter, conv_rows, input_cols, input_rows, filter_rows, input_channels):
		self.num_compute_clock_cycles_analog[self.current_layer] = self.batch_size * num_conv_in_input * col_fold * row_fold
		self.num_compute_clock_cycles_digital[self.current_layer] = -1
		self.num_program_compute_instance[self.current_layer] = row_fold * col_fold
		self.num_program_clock_cycles[self.current_layer] = -1

		self.SRAM_input_reads[self.current_layer] = self.batch_size * num_conv_in_input * ind_filter_size * col_fold
		self.SRAM_filter_reads[self.current_layer] = ind_filter_size * num_filter # this could be a problem, depends on order, right? 
		self.SRAM_output_writes[self.current_layer] = self.batch_size * num_conv_in_input *  num_filter * row_fold 
		self.SRAM_output_reads[self.current_layer] = self.SRAM_output_writes[self.current_layer]

		self.DRAM_filter_reads_analytical[self.current_layer] = ind_filter_size * num_filter
		self.DRAM_output_writes_analytical[self.current_layer] = self.SRAM_output_writes[self.current_layer]
		self.DRAM_output_reads_analytical[self.current_layer] = -1

		if (self.SRAM_input_size >= input_cols * input_rows):
			self.DRAM_input_reads_analytical[self.current_layer] = input_cols * input_rows
			print("not complicated situation for DRAM input reads")
		elif (self.SRAM_input_size >= filter_rows * input_cols):
			#self.DRAM_input_reads_analytical[self.current_layer] = -1
			num_cols_post_first_row_fill_SRAM = self.SRAM_input_size - filter_rows * input_cols 
			num_convs_fill_SRAM = num_cols_post_first_row_fill_SRAM + input_cols
			num_SRAM_fill = conv_rows * input_cols / num_convs_fill_SRAM # should this be input_rows instead? i don't think so
			num_complete_SRAM_fill = math.floor(num_SRAM_fill)   
			complete_SRAM_fill_accesses = num_complete_SRAM_fill * (self.SRAM_input_size)

			extra_cols = (num_SRAM_fill - num_complete_SRAM_fill) * num_convs_fill_SRAM
			if (extra_cols < input_cols):
				extra_SRAM_fill_accesses = extra_cols * filter_rows 
			else: 
				extra_SRAM_fill_accesses = input_cols * filter_rows + (extra_cols - input_cols) 
			
			self.DRAM_input_reads_analytical[self.current_layer] = round(extra_SRAM_fill_accesses + complete_SRAM_fill_accesses) * self.num_program_compute_instance[self.current_layer]
			print("most complicated situation for DRAM input reads")
		else: 
			self.DRAM_input_reads_analytical[self.current_layer] = conv_rows * input_cols * filter_rows * self.num_program_compute_instance[self.current_layer]
			print("not complicated situation for DRAM input reads")


		self.DRAM_input_reads_SRAM_sharing[self.current_layer] = self.DRAM_input_reads_analytical[self.current_layer] - self.SRAM_carryover_data_previous_layer
		self.DRAM_output_writes_SRAM_sharing[self.current_layer] = self.DRAM_output_writes_analytical[self.current_layer] - self.SRAM_carryover_data_current_layer
	
	def single_layer_set_params(self, NN_layer):
		input_rows  = NN_layer.loc["Input Rows"].item()
		input_cols  = NN_layer.loc["Input Columns"].item()
		filter_rows = NN_layer.loc["Filter Rows"].item()
		filter_cols = NN_layer.loc["Filter Columns"].item()
		channels    = NN_layer.loc["Channels"].item()
		num_filter  = NN_layer.loc["Num Filter"].item()
		xStride     = NN_layer.loc["X Stride"].item()
		yStride     = NN_layer.loc["Y Stride"].item()

		if (1):
			input_size = input_rows * input_cols * self.batch_size
			filter_size = filter_rows * filter_cols * num_filter * channels
			print("Input Size: ", input_size)
			print("Filter Size: ", filter_size)

		conv_rows = math.ceil((input_rows - filter_rows) / xStride) + 1 # math.ceil(input_rows / stride)
		conv_cols = math.ceil((input_cols - filter_cols) / yStride) + 1 # math.ceil(input_cols / stride)
		num_conv_in_input = conv_rows * conv_cols 
		ind_filter_size = filter_rows * filter_cols * channels

		col_fold = math.ceil(num_filter / self.array_cols)  
		row_fold = math.ceil(ind_filter_size / self.array_rows)

		#SRAM_input_output_crossover_data = 0
		#if ((self.current_layer != 0) and (self.SRAM_sharing)):
		#	SRAM_input_output_crossover_data = min(self.SRAM_output_size, self.SRAM_output_writes[self.current_layer - 1])

		self.compute_analytical_expressions(num_conv_in_input, col_fold, row_fold, ind_filter_size, num_filter, conv_rows, input_cols, input_rows, filter_rows, channels)
		
	
	def calculate_NN_totals(self):
		self.num_compute_clock_cycles_analog_total = sum(self.num_compute_clock_cycles_analog)
		self.num_compute_clock_cycles_digital_total = sum(self.num_compute_clock_cycles_digital)
		self.num_program_compute_instance_total = sum(self.num_program_compute_instance)
		self.num_program_clock_cycles_total = sum(self.num_program_clock_cycles)

		self.SRAM_input_reads_total   = sum(self.SRAM_input_reads)
		self.SRAM_filter_reads_total  = sum(self.SRAM_filter_reads)
		self.SRAM_output_writes_total = sum(self.SRAM_output_writes)
		self.SRAM_output_reads_total  = sum(self.SRAM_output_reads)

		self.DRAM_input_reads_analytical_total   = sum(self.DRAM_input_reads_analytical)
		self.DRAM_filter_reads_analytical_total  = sum(self.DRAM_filter_reads_analytical)
		self.DRAM_output_writes_analytical_total = sum(self.DRAM_output_writes_analytical)
		self.DRAM_output_reads_analytical_total  = sum(self.DRAM_output_reads_analytical)

		self.DRAM_input_reads_SRAM_sharing_total = sum(self.DRAM_input_reads_SRAM_sharing)
		self.DRAM_output_writes_SRAM_sharing_total = sum(self.DRAM_output_writes_SRAM_sharing)

		

	# note these have not been updated
	def print_NN_results(self):
		print("\n-----------Total Results Across all Layers-----------")
		print("Num Compute Clock Cycles Analog Total: ", self.num_compute_clock_cycles_analog_total)
		print("Num Compute Clock Cycles Digital Total:  ", self.num_compute_clock_cycles_digital_total)  
		print("Num Program Compute Instance Total: ", self.num_program_compute_instance_total)
		print("Num Program Clock Cycles Total: ", self.num_program_clock_cycles_total)
		print()

		print("SRAM Input Reads: ", self.SRAM_input_reads_total)
		print("SRAM Filter Reads: ", self.SRAM_filter_reads_total)
		print("SRAM Output Writes: ", self.SRAM_output_writes_total)
		print("SRAM Output Reads: ", self.SRAM_output_writes_total)
		print()

		print("DRAM Input Reads Analytical: ", self.DRAM_input_reads_analytical_total)
		print("DRAM Filter Reads Analytical: ", self.DRAM_filter_reads_analytical_total)
		print("DRAM Output Writes Analytical: ", self.DRAM_output_writes_analytical_total)
		print("DRAM Output Reads Analytical: ", self.DRAM_output_writes_analytical_total)
		print()

		print("DRAM Input Reads Simulation SRAM Sharing: ", self.DRAM_input_reads_SRAM_sharing_total)
		print("DRAM Output Writes Simulation SRAM Sharing: ", self.DRAM_output_writes_SRAM_sharing_total)
		#self.DRAM_output_writes_SRAM_sharing_total = sum(self.DRAM_output_writes_SRAM_sharing)



	def print_layer_results(self):
		for layer_num in range(self.num_NN_layers):
			print("\n----Results for layer", str(layer_num), "----")
			print("Num Compute Clock Cycles Analog: ", self.num_compute_clock_cycles_analog_total[layer_num])
			print("Num Compute Clock Cycles Digital Total:  ", self.num_compute_clock_cycles_digital_total[layer_num])  
			print("Num Program Compute Instance Total: ", self.num_program_compute_instance_total[layer_num])
			print("Num Program Clock Cycles Total: ", self.num_program_clock_cycles_total[layer_num])

			print("SRAM Input Reads: ", self.SRAM_input_reads[layer_num])
			print("SRAM Filter Reads: ", self.SRAM_filter_reads[layer_num])
			print("SRAM Output Writes: ", self.SRAM_output_writes[layer_num])
			print("DRAM Input Reads: ", self.DRAM_input_reads[layer_num])
			print("DRAM Filter Reads: ", self.DRAM_filter_reads[layer_num])
			print("DRAM Output Writes: ", self.DRAM_output_writes[layer_num])

	# but this has been updated
	def return_specs(self):
		runspecs_names = ["SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
			"DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes", \
			"Total Program/Compute Instances", "Total Programming Clock Cycles", \
			"Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital"]
	
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_total, \
			self.DRAM_input_reads_analytical_total, self.DRAM_filter_reads_analytical_total, self.DRAM_output_writes_analytical_total, \
			self.num_program_compute_instance_total, -1, \
			self.num_compute_clock_cycles_analog_total, -1]


		return(pd.DataFrame(totals, runspecs_names))
