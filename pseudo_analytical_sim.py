import math
import time
import SRAM_model
import pandas as pd
import numpy as np

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

		if (0):
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
		#print("\nModified Analytical Input DRAM Reads:", self.dram_input_reads_analytical, "\n")
		return(AM_results)

	def count_new_data(self, existing_data, demand_data):
		sum = 0
		if existing_data.shape != demand_data.shape:
			print("ERROR, MATRICES NOT SAME SIZE")
		else:
			for row in range(existing_data.shape[0]):
				for col in range(existing_data.shape[1]):
					if existing_data[row, col] == 0 and demand_data[row, col] == 1:
						sum += 1
		return sum

	def compute_input_DRAM_access(self):
		input_size = self.input_rows * self.input_cols
		if (self.SRAM_input_size >= input_size):
			self.DRAM_input_reads_analytical[self.current_layer] = input_size
			print("SRAM can fit entirety of input data")
		else:
			self.iterate_row_col_fold()
			self.DRAM_input_reads_analytical[self.current_layer] = round(self.DRAM_input_reads_analytical[self.current_layer])

	def make_local_conv_window_demand(self, row_fold_group):
		local_conv_window_demand = np.zeros([self.filter_rows, self.filter_cols])
		flattened_access_indices = np.arange(row_fold_group * self.array_rows, min(self.filter_rows * self.filter_cols, (row_fold_group + 1) * self.array_rows), 1)
		(row_ind, col_ind) = np.unravel_index(flattened_access_indices, [self.filter_rows, self.filter_cols])
		local_conv_window_demand[row_ind, col_ind] = 1
		return local_conv_window_demand


	def add_presence_points(self, conv_idx, local_conv_window_demand):
		### need to check if any presence points exist already or not, and if they do, ignore the index and put this thing at the starts
		conv_rows = math.ceil((self.input_rows - self.filter_rows) / self.x_stride) + 1 # math.ceil(self.input_rows / stride)
		conv_cols = math.ceil((self.input_cols - self.filter_cols) / self.y_stride) + 1 # math.ceil(self.input_cols / stride)
		(conv_row_idx, conv_col_idx) = np.unravel_index(int(conv_idx), [conv_rows, conv_cols])
		if (conv_row_idx != 0):
			presence_window_modified = np.zeros(local_conv_window_demand.shape)
			if self.y_stride < self.filter_rows:
				presence_window_modified[self.y_stride:self.filter_rows, :] = local_conv_window_demand[0:self.filter_rows - self.y_stride, :]
				if np.sum(presence_window_modified != 0):
					self.presence_change_indices = np.insert(self.presence_change_indices, 0, conv_idx - conv_rows)
					self.presence_windows.append(presence_window_modified)
		
		self.presence_change_indices = np.insert(self.presence_change_indices, 0, conv_idx)
		self.presence_windows.append(local_conv_window_demand)
		
		#if (conv_row_idx != conv_rows - 1):
		#	presence_change_indices = np.insert(presence_change_indices, 0, (conv_rows - 1) * conv_cols)
		#	presence_windows.append(local_conv_window_demand)
		
		args_sorted = np.argsort(self.presence_change_indices)[0:2]
		args_sorted = [1, 0, 2]
		self.presence_change_indices = self.presence_change_indices[args_sorted]
		#self.presence_windows = [self.presence_windows[i] for i in args_sorted]
		self.presence_windows = [self.presence_windows[0], self.presence_windows[1]]
		#return(presence_change_indices, presence_windows)

	def basic_operation_params(self):
		ind_filter_size = self.filter_rows * self.filter_cols * self.channels
		row_fold = math.ceil(ind_filter_size / self.array_rows)
		col_fold = math.ceil(self.num_filter / self.array_cols)  
		conv_rows = math.ceil((self.input_rows - self.filter_rows) / self.x_stride) + 1 # math.ceil(self.input_rows / stride)
		conv_cols = math.ceil((self.input_cols - self.filter_cols) / self.y_stride) + 1 # math.ceil(self.input_cols / stride)
		total_convs = conv_cols * conv_rows
		return(row_fold, col_fold, conv_rows, conv_cols, total_convs)
	
	def find_spot_in_presence_windows(self, conv_idx, first_row, conv_idx_leave_first_row):
		(_, _, _, conv_cols, _) = self.basic_operation_params()
		previous_presence_change_arg = max(np.argwhere(np.array(self.presence_change_indices) <= conv_idx))[0]
		current_presence_window = self.presence_windows[previous_presence_change_arg]
		next_presence_change = self.presence_change_indices[previous_presence_change_arg + 1]

		if first_row:
			if next_presence_change > conv_idx_leave_first_row:
				next_presence_change = conv_idx_leave_first_row

		return(next_presence_change, current_presence_window)
	
	def convs_min_overlap(self):
		return(math.floor((self.filter_rows - 1) / self.y_stride), math.floor((self.filter_cols - 1) / self.x_stride))
	
	def make_embedded_presence_array(self, current_presence_window, local_conv_window_demand, first_row_input):
		(_, extra_conv_cols_single_side) = self.convs_min_overlap() 
		num_rows = self.y_stride + self.filter_rows; num_cols = extra_conv_cols_single_side * 2 * self.x_stride + self.filter_cols
		test_array = np.zeros([num_rows, num_cols])
		# fill with the local conv demand in the top row
		for row in range(0, self.y_stride + 1, self.y_stride):
			for col in range(0, num_cols - self.filter_cols + 1, self.x_stride):
				test_array_indices = tuple([slice(row, row + self.filter_rows), slice(col, col + self.filter_cols)])
				if row is 0:
					if not first_row_input:
						# put down lcoal conv window demand
						test_array[test_array_indices] = np.logical_or(test_array[test_array_indices], local_conv_window_demand)
				else:
					test_array[test_array_indices] = np.logical_or(test_array[test_array_indices], current_presence_window)
		return(test_array)
	
	def traverse_embedded_presence_with_demand(self, embedded_presence, local_conv_window_demand):
		first_col_new_data = 0; ho_movement_new_data = 0
		total_data_row = 0; extra_data_accumulated = 0

		extra_conv_rows = math.floor((self.filter_rows - 1) / self.y_stride) * 2

		row = (extra_conv_rows / 2) * self.y_stride; col = 0; col_count = 0
		for col in range(0, embedded_presence.shape[1] - self.filter_cols + 1, self.x_stride):
			test_array_indices = tuple([slice(row, row + self.filter_rows), slice(col, col + self.filter_cols)])
			data_in_current_conv_window = embedded_presence[test_array_indices]
			col_count += 1
			new_data_count = self.count_new_data(data_in_current_conv_window, local_conv_window_demand) 
			total_data_row += new_data_count
			if col == 0:
				first_col_new_data = new_data_count
			embedded_presence[test_array_indices] = np.logical_or(embedded_presence[test_array_indices], local_conv_window_demand)
		ho_movement_new_data = new_data_count 
		
				#col += self.x_stride
		extra_data_accumulated = total_data_row - (col_count - 1) * ho_movement_new_data - first_col_new_data
		return(first_col_new_data, ho_movement_new_data, total_data_row, extra_data_accumulated)
	
		
	def local_conv_window_basic_movements(self, local_conv_window_demand, conv_idx, first_row, conv_idx_leave_first_row):
		(next_presence_change, current_presence_window) = self.find_spot_in_presence_windows(conv_idx, first_row, conv_idx_leave_first_row)
		embedded_presence_array = self.make_embedded_presence_array(current_presence_window, local_conv_window_demand, first_row)
		return((self.traverse_embedded_presence_with_demand(embedded_presence_array, local_conv_window_demand), next_presence_change))
		#return(eff_local_demand_window_size, new_data_per_ho_movement_first_row, new_data_per_vert_movement_first_col, new_data_per_ho_movement_later_row, extra_data_end_of_first_row, extra_data_end_of_later_row)	

	def iterate_row_col_fold(self):
		#### ***** still need to add first row extras 
		def calculate_convs_to_fill_SRAM():
			return(effective_SRAM_size / average_new_data_added)

		def manage_conv_target_overreach(conv_target, start_conv_idx):
			nonlocal first_row, effective_SRAM_size, conv_idx
			if conv_target == conv_idx_leave_first_row:
				first_row = 0
				
			remaining_convs = conv_target - start_conv_idx
			remaining_data_reads = remaining_convs * average_new_data_added
			
			self.DRAM_input_reads_analytical[self.current_layer] += remaining_data_reads
			conv_idx = conv_target
			effective_SRAM_size -= remaining_data_reads
		
		def manage_full_SRAM():
			nonlocal conv_idx, effective_SRAM_size, conv_idx_last_SRAM_fill, conv_idx_leave_first_row, first_row
			conv_idx += convs_fill_SRAM
			self.DRAM_input_reads_analytical[self.current_layer] += effective_SRAM_size
			effective_SRAM_size = self.SRAM_input_size
			conv_idx_last_SRAM_fill = conv_idx
			conv_idx_leave_first_row = min(total_convs, conv_idx + conv_cols)
			first_row = 1
			reset_presence_data()

		def reset_presence_data():
			self.presence_change_indices = [0, total_convs] # np.array([0, total_convs]); 
			num_final_rows = math.floor((self.filter_rows - 1) / self.y_stride)
			for row in range(num_final_rows):
				self.presence_change_indices.append((conv_rows - row - 1) * conv_cols)
			#self.presence_change_indices.append(conv_cols)
			self.presence_windows = [np.zeros([self.filter_rows, self.filter_cols])] * len(self.presence_change_indices)
			self.presence_change_indices.sort()

		(row_fold, col_fold, conv_rows, conv_cols, total_convs) = self.basic_operation_params()
		conv_idx = 0; first_row = 1; effective_SRAM_size = self.SRAM_input_size; conv_idx_last_SRAM_fill = 0; conv_idx_leave_first_row = conv_cols
		reset_presence_data()
		
		for col_fold_group in range(col_fold):
			for row_fold_group in range(row_fold):
				local_conv_window_demand = self.make_local_conv_window_demand(row_fold_group)
				conv_idx = 0; first_row = 1; conv_idx_leave_first_row = min(total_convs, conv_cols)
				while (conv_idx < total_convs):
					(average_new_data_added, conv_idx_next_presence_change) = \
						self.local_conv_window_basic_movements(local_conv_window_demand, conv_idx, first_row, conv_idx_leave_first_row)

					convs_fill_SRAM = calculate_convs_to_fill_SRAM()
					if conv_idx + convs_fill_SRAM > conv_idx_next_presence_change:
						manage_conv_target_overreach(conv_idx_next_presence_change, conv_idx)					
						if conv_idx_next_presence_change == total_convs: 
							#self.add_presence_points(conv_idx_last_SRAM_fill, local_conv_window_demand)
							conv_idx_next_presence_change = conv_cols
					else: 
						manage_full_SRAM()

					
	def single_layer_set_params(self, NN_layer):
		self.input_rows  = NN_layer.loc["Input Rows"].item()
		self.input_cols  = NN_layer.loc["Input Columns"].item()
		self.filter_rows = NN_layer.loc["Filter Rows"].item()
		self.filter_cols = NN_layer.loc["Filter Columns"].item()
		self.channels    = NN_layer.loc["Channels"].item()
		self.num_filter  = NN_layer.loc["Num Filter"].item()
		self.x_stride     = NN_layer.loc["X Stride"].item()
		self.y_stride     = NN_layer.loc["Y Stride"].item()

		if (0):
			input_size = self.input_rows * self.input_cols * self.batch_size
			filter_size = self.filter_rows * self.filter_cols * self.num_filter * self.channels
			print("Input Size: ", input_size)
			print("Filter Size: ", filter_size)

		conv_rows = math.ceil((self.input_rows - self.filter_rows) / self.x_stride) + 1 # math.ceil(self.input_rows / stride)
		conv_cols = math.ceil((self.input_cols - self.filter_cols) / self.y_stride) + 1 # math.ceil(self.input_cols / stride)
		num_conv_in_input = conv_rows * conv_cols 
		ind_filter_size = self.filter_rows * self.filter_cols * self.channels

		col_fold = math.ceil(self.num_filter / self.array_cols)  
		row_fold = math.ceil(ind_filter_size / self.array_rows)

		if ((conv_cols - 1) * self.x_stride + self.filter_cols != self.input_cols):
			print("ERROR. X STRIDE NOT SAME ALL THE WAY ACROSS")
			print("Input Cols:", self.input_cols)
			print("Better number of input cols: ", (conv_cols - 1) * self.x_stride + self.filter_cols)
		else: print("OK number of cols based on x stride")

		if ((conv_rows - 1) * self.y_stride + self.filter_rows != self.input_rows):
			print("ERROR. Y STRIDE NOT SAME ALL THE WAY ACROSS")
			print("Input Rows:", self.input_rows)
			print("Better number of input rows: ", (conv_rows - 1) * self.y_stride + self.filter_rows)
		else: print("OK number of rows based on y stride")

		self.num_compute_clock_cycles_analog[self.current_layer] = self.batch_size * num_conv_in_input * col_fold * row_fold
		self.num_compute_clock_cycles_digital[self.current_layer] = -1
		self.num_program_compute_instance[self.current_layer] = row_fold * col_fold
		self.num_program_clock_cycles[self.current_layer] = -1

		self.SRAM_input_reads[self.current_layer] = self.batch_size * num_conv_in_input * ind_filter_size * col_fold
		self.SRAM_filter_reads[self.current_layer] = ind_filter_size * self.num_filter # this could be a problem, depends on order, right? 
		self.SRAM_output_writes[self.current_layer] = self.batch_size * num_conv_in_input *  self.num_filter * row_fold 
		self.SRAM_output_reads[self.current_layer] = self.SRAM_output_writes[self.current_layer]

		self.DRAM_filter_reads_analytical[self.current_layer] = ind_filter_size * self.num_filter
		self.DRAM_output_writes_analytical[self.current_layer] = self.SRAM_output_writes[self.current_layer]
		self.DRAM_output_reads_analytical[self.current_layer] = -1

		self.DRAM_input_reads_SRAM_sharing[self.current_layer] = -1#self.DRAM_input_reads_analytical[self.current_layer] - self.SRAM_carryover_data_previous_layer
		self.DRAM_output_writes_SRAM_sharing[self.current_layer] = -1# self.DRAM_output_writes_analytical[self.current_layer] - self.SRAM_carryover_data_current_layer
	
		#SRAM_input_output_crossover_data = 0
		#if ((self.current_layer != 0) and (self.SRAM_sharing)):
		#	SRAM_input_output_crossover_data = min(self.SRAM_output_size, self.SRAM_output_writes[self.current_layer - 1])
		self.compute_input_DRAM_access()

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
