import math
import time
#from regex import B
import SRAM_model
import pandas as pd
num_conv_in_input_delete = 0
#import sim_params_analytical
#import specs_info


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
		self.num_program_instance = [0] * self.num_NN_layers
		self.num_compute_instance = [0] * self.num_NN_layers
		#self.num_programming_theory   = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_analog = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_digital = [0] * self.num_NN_layers
		self.num_program_clock_cycles = [0] * self.num_NN_layers
		#self.num_compute_cycles_theory = [0] * self.num_NN_layers

		self.SRAM_input_reads = [0] * self.num_NN_layers
		self.SRAM_filter_reads = [0] * self.num_NN_layers
		self.SRAM_output_writes = [0] * self.num_NN_layers
		self.SRAM_output_reads = [0] * self.num_NN_layers

		self.DRAM_input_reads = [0] * self.num_NN_layers
		self.DRAM_filter_reads = [0] * self.num_NN_layers
		self.DRAM_output_writes = [0] * self.num_NN_layers
		self.DRAM_output_reads = [0] * self.num_NN_layers

		self.SRAM_DRAM_input_misses = [0] * self.num_NN_layers
		self.SRAM_DRAM_filter_misses = [0] * self.num_NN_layers
		self.SRAM_DRAM_output_misses = [0] * self.num_NN_layers

		self.SRAM_input_accesses = [0] * self.num_NN_layers
		self.SRAM_filter_accesses = [0] * self.num_NN_layers
		self.SRAM_output_accesses = [0] * self.num_NN_layers


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
			status = self.single_layer_set_params(layer)
			if (status == -1):
				return
		#print()
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

	def single_layer_set_params(self, NN_layer):
		input_rows  = NN_layer["Input Rows"]
		input_cols  = NN_layer["Input Columns"]
		filter_rows = NN_layer["Filter Rows"]
		filter_cols = NN_layer["Filter Columns"]
		channels    = NN_layer["Channels"]
		num_filter  = NN_layer["Num Filter"]
		xStride     = NN_layer["X Stride"]
		yStride     = NN_layer["Y Stride"]



		input_divider = 2
		col_divider_for_output = 2

		filter_size = filter_rows * filter_cols * channels
		input_block_size = round(input_cols * input_rows * channels / input_divider)

		col_fold = math.ceil(num_filter / self.array_cols)  
		row_fold = math.ceil(filter_size / self.array_rows)
		#input_fold = math.ceil(self.batch_size * input_divider)
		input_fold = math.ceil(input_cols * input_rows * channels * self.batch_size / input_block_size)

		conv_rows = math.ceil((input_rows - filter_rows) / xStride) + 1
		conv_cols = math.ceil((input_cols - filter_cols) / yStride) + 1
		#conv_rows = math.ceil(input_rows / stride)
		#conv_cols = math.ceil(input_cols / stride)

		num_conv_in_input = conv_rows * conv_cols 
		#global num_conv_in_input_delete 
		#num_conv_in_input_delete = num_conv_in_input

		self.num_compute_cycles_theory[self.current_layer] = self.batch_size *  num_conv_in_input * col_fold * row_fold 
		self.SRAM_input_reads[self.current_layer] = self.batch_size * num_conv_in_input * filter_size * col_fold

		self.SRAM_filter_reads[self.current_layer] = filter_size * num_filter # this could be a problem, depends on order, right? 
		# like need to multiply by batch size if input isn't on the very inside 


		self.SRAM_output_writes[self.current_layer] = num_conv_in_input * self.batch_size * num_filter * row_fold 
		self.SRAM_output_reads[self.current_layer] = num_conv_in_input * self.batch_size * num_filter * row_fold

		if self.current_layer == 0:
			SRAM_input_output_crossover_data = 0
		else:
			SRAM_input_output_crossover_data = min(self.SRAM_output_size, self.SRAM_output_writes[self.current_layer - 1])

		new_mem = self.input_SRAM.new_layer(input_block_size, input_fold, SRAM_input_output_crossover_data)
		if new_mem == -1:
			return -1
		new_mem = self.filter_SRAM.new_layer(self.array_cols * self.array_rows, row_fold * col_fold, 0)
		if new_mem == -1:
			return -1
		new_mem = self.output_SRAM.new_layer(num_conv_in_input * self.array_cols / col_divider_for_output, col_fold * col_divider_for_output, 0)
		if new_mem == -1:
			return -1


		self.run_single_layer(col_fold, row_fold, batch_fold)
		#if (self.num_programming_theory != self.num_programming_practice):
		#	print("Theoretical Programming Count NOT EQUAL to Practical Programming Count")
		#	return(-1)


		if self.current_layer != (self.num_NN_layers - 1): 
			self.DRAM_output_writes[self.current_layer] -= min(self.SRAM_input_size, self.SRAM_output_writes[self.current_layer])  # note that if this thing isn't very full we don't save very much 



			max(self.SRAM_output_writes[self.current_layer] - self.SRAM_output_size, 0)

		self.input_SRAM.conclude_layer()
		self.filter_SRAM.conclude_layer()
		self.output_SRAM.conclude_layer()

	def run_single_layer(self, col_fold, row_fold, input_fold = 1):     
		#self.num_programming_practice[self.current_layer] = 0
		#self.num_programming_theory[self.current_layer] = col_fold * row_fold * batch_fold
		#if (row_fold == 1) : # do we still need this? 
		#	self.num_programming_theory[self.current_layer] = col_fold

		global print_string
		old_col_group = -1
		old_row_group = -1

		num_loop_iterations = col_fold * row_fold * batch_fold
		current_loop_iteration = 0
		target_inc = 10
		target = target_inc
		for col_group in range(col_fold):
			for row_group in range(row_fold):
				for input_group in range(input_fold):
					current_loop_iteration += 1
					percent_done = 100 * current_loop_iteration / num_loop_iterations
					if percent_done >= target:
						addon = "percent done: "+ str(round(percent_done)) + "  "
						print_string += addon
						print(print_string, end="\r", flush=True)
						print_string = print_string[0:len(print_string) - len(addon)]
						target += target_inc

						#if (old_col_group != current_col_group) or (old_row_group != current_row_group):
						#		self.num_programming_practice[self.current_layer] += 1
						#self.num_compute_cycles_practice[self.current_layer] += num_conv_in_input_delete 

						#old_col_group = col_group
						#old_row_group = row_group

						if (0):
							print("")
							print("Current Col Group:", col_group)
							print("Current Row Group:", row_group)
							print("Current Batch Group:", batch)

						filter_index = col_group * row_fold + row_group
						self.manage_SRAM_DRAM_access(current_batch, filter_index, col_fold)

	#self.SRAM_filter_reads[self.current_layer] = self.num_programming_practice[self.current_layer] * self.
		

	def manage_SRAM_DRAM_access(self, input_index, filter_index, output_index):
		#start_time = time.time()
		self.input_SRAM.access_component(current_batch)
		#med_time = time.time()
		self.filter_SRAM.access_component(filter_index)
		self.output_SRAM.access_component(output_index)
		#end_time = time.time()
		#global input_sram_time, filter_sram_time
		#input_sram_time += med_time - start_time
		#filter_sram_time += end_time - med_time

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

	def return_specs(self):
		runspecs_names = ["SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
			"DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes",\
				"Total Programming Instances", "Total Programming Clock Cycles", \
				"Total Compute Instances", "Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital"]
		
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_total, \
					self.DRAM_input_reads_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_total, \
					self.num_programming_theory_total, -1, \
						-1, self.num_compute_cycles_theory_total, -1]
		

		return(pd.DataFrame(totals, runspecs_names))
