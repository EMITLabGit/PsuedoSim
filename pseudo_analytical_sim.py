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
		self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"].item()
		self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"].item()
		self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"].item()
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

		self.DRAM_input_reads = [0] * self.num_NN_layers
		self.DRAM_filter_reads = [0] * self.num_NN_layers
		self.DRAM_output_writes = [0] * self.num_NN_layers
		self.DRAM_output_reads = [0] * self.num_NN_layers

		self.DRAM_input_reads_total   = 0
		self.DRAM_filter_reads_total  = 0
		self.DRAM_output_writes_total = 0
		self.DRAM_output_reads_total  = 0



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
			print(print_string, end="\r", flush=True)
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
		print("Done with simulation, it took", AM_execution_time, "minutes                          ")
		self.access_SRAM_data()
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
	

	def compute_analytical_expressions(self, num_conv_in_input, col_fold, row_fold, ind_filter_size, num_filter):
		self.num_compute_clock_cycles_analog[self.current_layer] = self.batch_size * num_conv_in_input * col_fold * row_fold
		self.num_compute_clock_cycles_digital[self.current_layer] = -1
		self.num_program_compute_instance[self.current_layer] = row_fold * col_fold
		self.num_program_clock_cycles[self.current_layer] = -1

		self.SRAM_input_reads[self.current_layer] = self.batch_size * num_conv_in_input * ind_filter_size * col_fold
		self.SRAM_filter_reads[self.current_layer] = ind_filter_size * num_filter # this could be a problem, depends on order, right? 
		self.SRAM_output_writes[self.current_layer] = num_conv_in_input * self.batch_size * num_filter * row_fold 
		self.SRAM_output_reads[self.current_layer] = self.SRAM_output_writes[self.current_layer]

	def set_SRAM_modules(self, input_block_size, input_block_fold, filter_block_size, filter_block_fold, output_block_size, output_block_fold):
		new_mem = self.input_SRAM.new_layer(input_block_size, input_block_fold, 0)
		if new_mem == -1:
			return 0
		new_mem = self.filter_SRAM.new_layer(filter_block_size, filter_block_fold, 0)
		if new_mem == -1:
			return 0
		new_mem = self.output_SRAM.new_layer(output_block_size, output_block_fold, 0)
		if new_mem == -1:
			return 0
		
		return 1
	
	def manage_final_layer_DRAM(self, total_output_size):
		self.DRAM_input_reads[self.current_layer] = self.input_SRAM.DRAM_reads_layer
		self.DRAM_filter_reads[self.current_layer] = self.filter_SRAM.DRAM_reads_layer
		self.DRAM_output_writes[self.current_layer] = self.output_SRAM.DRAM_reads_layer
		self.DRAM_input_reads[self.current_layer] = self.DRAM_output_writes[self.current_layer] - total_output_size


	def single_layer_set_params(self, NN_layer):
		input_rows  = NN_layer.loc["Input Rows"].item()
		input_cols  = NN_layer.loc["Input Columns"].item()
		filter_rows = NN_layer.loc["Filter Rows"].item()
		filter_cols = NN_layer.loc["Filter Columns"].item()
		channels    = NN_layer.loc["Channels"].item()
		num_filter  = NN_layer.loc["Num Filter"].item()
		xStride     = NN_layer.loc["X Stride"].item()
		yStride     = NN_layer.loc["Y Stride"].item()

		input_divider = 2

		conv_rows = math.ceil((input_rows - filter_rows) / xStride) + 1 # math.ceil(input_rows / stride)
		conv_cols = math.ceil((input_cols - filter_cols) / yStride) + 1 # math.ceil(input_cols / stride)
		num_conv_in_input = conv_rows * conv_cols 
		ind_filter_size = filter_rows * filter_cols * channels

		total_input_size = input_cols * input_rows * channels * self.batch_size
		input_block_size = round(total_input_size / (self.batch_size * input_divider))
		input_block_fold = math.ceil(total_input_size / input_block_size)

		col_fold = math.ceil(num_filter / self.array_cols)  
		row_fold = math.ceil(ind_filter_size / self.array_rows)
		filter_block_size = self.array_rows * self.array_cols
		filter_block_fold = row_fold * col_fold

		total_output_size = num_conv_in_input * self.batch_size * num_filter
		output_block_fold = input_block_fold * col_fold
		output_block_size = math.ceil(total_output_size / output_block_fold)

		print("Input  block size, fold: ", input_block_size, ", ", input_block_fold, sep = "")
		print("Filter block size, fold: ", filter_block_size, ", ", filter_block_fold, sep = "")
		print("Output block size, fold: ", output_block_size, ", ", output_block_fold, sep = "")


		#SRAM_input_output_crossover_data = 0
		#if ((self.current_layer != 0) and (self.SRAM_sharing)):
		#	SRAM_input_output_crossover_data = min(self.SRAM_output_size, self.SRAM_output_writes[self.current_layer - 1])

		self.compute_analytical_expressions(num_conv_in_input, col_fold, row_fold, ind_filter_size, num_filter)
		if not self.set_SRAM_modules(input_block_size, input_block_fold, filter_block_size, filter_block_fold, output_block_size, output_block_fold):
			return -1

		self.SRAM_carryover_data_current_layer = min(self.SRAM_output_size, min(self.SRAM_input_size, self.SRAM_output_writes[self.current_layer]))
		if (self.current_layer == self.num_NN_layers - 1): 
			self.SRAM_carryover_data_current_layer = 0

		self.run_single_layer(col_fold, row_fold, input_block_fold)		
		self.manage_final_layer_DRAM(total_output_size)
		self.SRAM_carryover_data_previous_layer = self.SRAM_carryover_data_current_layer
		
		# ultimately want to make this part of the SRAM module
		#if self.current_layer != (self.num_NN_layers - 1): 
		#	self.DRAM_output_writes[self.current_layer] -= min(self.SRAM_input_size, self.SRAM_output_writes[self.current_layer])  # note that if this thing isn't very full we don't save very much 


	def run_single_layer(self, col_fold, row_fold, input_fold):     
		global print_string

		num_loop_iterations = col_fold * row_fold * input_fold
		current_loop_iteration = 0
		target_inc = 10
		target = target_inc
		for col_block in range(col_fold):
			for row_block in range(row_fold):
				for input_block in range(input_fold):
					#ultimately should make this a separate function 
					if (0):
						current_loop_iteration += 1
						percent_done = 100 * current_loop_iteration / num_loop_iterations
						if percent_done >= target:
							addon = "percent done: "+ str(round(percent_done)) + "  "
							print_string += addon
							print(print_string, end="\r", flush=True)
							print_string = print_string[0:len(print_string) - len(addon)]
							target += target_inc

					filter_block = col_block * row_fold   + row_block
					output_block = col_block * input_fold + input_block
					if (1):
						print("Current col block: ", col_block)
						print("Current row block: ", row_block)
						print("Current input  block: ", input_block)
						print("Current filter block: ", filter_block)
						print("Current output block: ", output_block)
						print()

					self.manage_SRAM_DRAM_access(input_block, filter_block, output_block)		

	def manage_SRAM_DRAM_access(self, input_index, filter_index, output_index):
		#start_time = time.time()
		self.input_SRAM.access_component(input_index)
		#med_time = time.time()
		self.filter_SRAM.access_component(filter_index)
		self.output_SRAM.access_component(output_index)
		#end_time = time.time()
		#global input_sram_time, filter_sram_time
		#input_sram_time += med_time - start_time
		#filter_sram_time += end_time - med_time

	def access_SRAM_data(self):
		#self.DRAM_input_reads  = self.input_SRAM.DRAM_reads
		#self.DRAM_filter_reads = self.filter_SRAM.DRAM_reads
		#self.DRAM_output_writes = self.output_SRAM.DRAM_reads

		self.input_SRAM.conclude_NN()
		self.filter_SRAM.conclude_NN()
		self.output_SRAM.conclude_NN()

	def calculate_NN_totals(self):
		self.num_compute_clock_cycles_analog_total = sum(self.num_compute_clock_cycles_analog)
		self.num_compute_clock_cycles_digital_total = sum(self.num_compute_clock_cycles_digital)
		self.num_program_compute_instance_total = sum(self.num_program_compute_instance)
		self.num_program_clock_cycles_total = sum(self.num_program_clock_cycles)

		self.SRAM_input_reads_total   = sum(self.SRAM_input_reads)
		self.SRAM_filter_reads_total  = sum(self.SRAM_filter_reads)
		self.SRAM_output_writes_total = sum(self.SRAM_output_writes)
		self.SRAM_output_reads_total  = sum(self.SRAM_output_reads)

		self.DRAM_input_reads_total   = sum(self.DRAM_input_reads)
		self.DRAM_filter_reads_total  = sum(self.DRAM_filter_reads)
		self.DRAM_output_writes_total = sum(self.DRAM_output_writes)
		self.DRAM_output_reads_total  = sum(self.DRAM_output_reads)


	def print_NN_results(self):
		print("\n-----------Total Results Across all Layers-----------")
		print("Num Compute Clock Cycles Analog Total: ", self.num_compute_clock_cycles_analog_total)
		print("Num Compute Clock Cycles Digital Total:  ", self.num_compute_clock_cycles_digital_total)  
		print("Num Program Compute Instance Total: ", self.num_program_compute_instance_total)
		print("Num Program Clock Cycles Total: ", self.num_program_clock_cycles_total)

		print("SRAM Input Reads: ", self.SRAM_input_reads_total)
		print("SRAM Filter Reads: ", self.SRAM_filter_reads_total)
		print("SRAM Output Writes: ", self.SRAM_output_writes_total)
		print("DRAM Input Reads: ", self.DRAM_input_reads_total)
		print("DRAM Filter Reads: ", self.DRAM_filter_reads_total)
		print("DRAM Output Writes: ", self.DRAM_output_writes_total)

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


	def return_specs(self):
		runspecs_names = ["SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
			"DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes",\
				"Total Program/Compute Instances", "Total Programming Clock Cycles", \
				"Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital"]
		
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_total, \
					self.DRAM_input_reads_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_total, \
					self.num_program_compute_instance_total, -1, \
						self.num_compute_clock_cycles_analog_total, -1]
		

		return(pd.DataFrame(totals, runspecs_names))
