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

	def add_to_text_output(self, string_output):
		self.text_output += "\n" + string_output

	def set_hardware(self, hardware_state_info):
		self.array_rows = hardware_state_info.loc["Systolic Array Rows"]#.item()
		self.array_cols = hardware_state_info.loc["Systolic Array Cols"]#.item()
		self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"] * 1000 / 2 #.item() * 1000 / 2
		self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"] * 1000 / 2 #.item() * 1000 / 2
		self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"] * 1000 / 2 #.item() * 1000 / 2
		self.batch_size = hardware_state_info.loc["Batch Size"]#.item()

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

		self.text_output = ""

	def set_NN(self, NN_layers_all):
		self.NN_layers_all = NN_layers_all
		self.num_NN_layers = len(NN_layers_all)

	def set_results_vars(self):
		self.num_program_compute_instance     = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_analog  = [0] * self.num_NN_layers
		self.num_compute_clock_cycles_digital = [0] * self.num_NN_layers
		self.num_program_clock_cycles         = [0] * self.num_NN_layers

		self.SRAM_input_reads      = [0] * self.num_NN_layers
		self.SRAM_filter_reads     = [0] * self.num_NN_layers
		self.SRAM_output_writes_SS = [0] * self.num_NN_layers

		self.DRAM_filter_reads     = [0] * self.num_NN_layers 
		self.DRAM_output_writes_SS = [0] * self.num_NN_layers

		self.DRAM_input_reads_analog  = [0] * self.num_NN_layers
		self.DRAM_input_reads_digital = [0] * self.num_NN_layers

		self.SRAM_output_writes_acc = [0] * self.num_NN_layers
		self.SRAM_output_reads_acc  = [0] * self.num_NN_layers
		self.DRAM_output_reads_acc  = [0] * self.num_NN_layers
		self.DRAM_output_writes_acc = [0] * self.num_NN_layers

		self.DRAM_input_reads_digital_SRAM_sharing = [0] * self.num_NN_layers
		self.DRAM_output_writes_acc_SRAM_sharing   = [0] * self.num_NN_layers

	def run_all_layers(self):
		#print("\nBeginning analytical modeling simulation")
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
		AM_post_process_time = round((final_time - end_time) / 60, 5)
		#self.print_layer_results()
		#self.print_NN_results()
		#self.save_all_layers_csv()
		AM_results = self.return_specs()
		AM_results.loc[" "] = " "
		AM_results.loc["Simulation Run Time [min]"] = AM_execution_time
		AM_results.loc["Simulation Post Process Time [min]"] = AM_post_process_time
		#print("\nModified Analytical Input DRAM Reads:", self.dram_input_reads, "\n")
		return(AM_results, self.text_output)

	def count_new_data(self, existing_data, demand_data):
		sum = 0
		if existing_data.shape != demand_data.shape:
			print("ERROR, MATRICES NOT SAME SIZE")
		else:
			for row in range(existing_data.shape[0]):
				for col in range(existing_data.shape[1]):
					for channel in range(existing_data.shape[2]):
						if existing_data[row, col, channel] == 0 and demand_data[row, col, channel] == 1:
							sum += 1
		return sum

	def compute_input_DRAM_access(self):
		input_size = self.input_rows * self.input_cols * self.channels
		if (self.SRAM_input_size >= input_size):
			self.DRAM_input_reads_analog[self.current_layer] = input_size
			self.add_to_text_output("SRAM can fit entirety of input data")
		else:
			self.add_to_text_output("SRAM canNOT fit entirety of input data")
			self.iterate_row_col_fold()
			self.DRAM_input_reads_analog[self.current_layer] = round(self.DRAM_input_reads_analog[self.current_layer])

	def make_local_conv_window_demand(self, row_fold_group):
		local_conv_window_demand = np.zeros([self.filter_rows, self.filter_cols, self.channels])
		flattened_access_indices = np.arange(row_fold_group * self.array_rows, min(self.filter_rows * self.filter_cols * self.channels, (row_fold_group + 1) * self.array_rows), 1)
		(row_ind, col_ind, channel_ind) = np.unravel_index(flattened_access_indices, local_conv_window_demand.shape)
		local_conv_window_demand[row_ind, col_ind, channel_ind] = 1
		return local_conv_window_demand

	def add_presence_points(self, conv_idx_last_SRAM_fill, local_conv_window_demand):
		def single_point_entry(new_change_start_conv_idx, new_change_end_conv_idx, shifted_presence_window):
			#new_change_conv_idx += 2
			new_change_list_idx = max(np.argwhere(self.presence_change_indices <= new_change_start_conv_idx))[0]
			if self.presence_change_indices[new_change_list_idx] == new_change_start_conv_idx:
				self.presence_windows[new_change_list_idx] = np.logical_or(self.presence_windows[new_change_list_idx], shifted_presence_window)
			else: 
				new_change_list_idx += 1
				self.presence_change_indices = np.insert(self.presence_change_indices, new_change_list_idx, new_change_start_conv_idx)
				self.presence_windows.insert(new_change_list_idx, np.logical_or(shifted_presence_window, self.presence_windows[new_change_list_idx - 1]))
			
			for presence_change_list_idx in range(new_change_list_idx + 1, len(self.presence_windows)):
				if self.presence_change_indices[presence_change_list_idx] > new_change_end_conv_idx: break
				self.presence_windows[presence_change_list_idx] = np.logical_or(self.presence_windows[presence_change_list_idx], shifted_presence_window) 
			
		(_, _, _, conv_cols, total_convs) = self.basic_operation_params()
		(extra_conv_rows_single_side, _) = self.convs_min_overlap()
		rows = self.filter_rows
		for row_shift in range(-extra_conv_rows_single_side * self.x_stride, extra_conv_rows_single_side * self.x_stride + 1, self.x_stride):
			shifted_presence_start_idx = max(conv_idx_last_SRAM_fill + (row_shift * conv_cols) / self.x_stride, 0) # if greater than total convs, no go 
			shifted_presence_end_idx   = min(total_convs + (row_shift * conv_cols) / self.x_stride - 1, total_convs)
			#if shifted_presence_change_idx < 0: shifted_presence_change_idx = 0
			if shifted_presence_start_idx < total_convs:
				presence_window_shifted = np.zeros(local_conv_window_demand.shape)
				presence_window_shifted[max(0, -row_shift):min(self.filter_rows - row_shift, self.filter_rows), :, :] = local_conv_window_demand[max(0, row_shift):min(self.filter_rows + row_shift, self.filter_rows), :, :]
				if (np.sum(presence_window_shifted) == 0): continue
				single_point_entry(shifted_presence_start_idx, shifted_presence_end_idx, presence_window_shifted)			
				#print(presence_window_modified)

	def basic_operation_params(self):
		ind_filter_size = self.filter_rows * self.filter_cols * self.channels
		row_fold = math.ceil(ind_filter_size / self.array_rows)
		col_fold = math.ceil(self.num_filter / self.array_cols)  
		conv_rows = math.ceil((self.input_rows - self.filter_rows) / self.x_stride) + 1 # math.ceil(self.input_rows / stride)
		conv_cols = math.ceil((self.input_cols - self.filter_cols) / self.y_stride) + 1 # math.ceil(self.input_cols / stride)
		total_convs = conv_cols * conv_rows
		return(row_fold, col_fold, conv_rows, conv_cols, total_convs)
	
	def find_spot_in_presence_windows(self, conv_idx, first_row, conv_idx_leave_first_row):
		(_, _, _, _, total_convs) = self.basic_operation_params()
		previous_presence_change_arg = max(np.argwhere(self.presence_change_indices <= conv_idx))[0]
		current_presence_window = self.presence_windows[previous_presence_change_arg]
		if previous_presence_change_arg == len(self.presence_windows) - 1:
			next_presence_change = total_convs
		else: next_presence_change = self.presence_change_indices[previous_presence_change_arg + 1]

		if first_row and next_presence_change > conv_idx_leave_first_row:
			next_presence_change = conv_idx_leave_first_row

		return(next_presence_change, current_presence_window)
	
	def convs_min_overlap(self):
		return(math.floor((self.filter_rows - 1) / self.y_stride), math.floor((self.filter_cols - 1) / self.x_stride))
	
	def make_embedded_presence_array(self, current_presence_window, local_conv_window_demand, first_row_input):
		current_presence_empty = np.sum(current_presence_window) == 0
		(_, extra_conv_cols_single_side) = self.convs_min_overlap() 
		num_rows = self.y_stride + self.filter_rows; num_cols = extra_conv_cols_single_side * 2 * self.x_stride + self.filter_cols
		test_array = np.zeros([num_rows, num_cols, self.channels])
		# fill with the local conv demand in the top row
		for row in range(0, self.y_stride + 1, self.y_stride):
			for col in range(0, num_cols - self.filter_cols + 1, self.x_stride):
				test_array_indices = tuple([slice(row, row + self.filter_rows), slice(col, col + self.filter_cols), slice(0, self.channels)])
				if row == 0:
					if not first_row_input:
						# put down lcoal conv window demand
						test_array[test_array_indices] = np.logical_or(test_array[test_array_indices], local_conv_window_demand)
				else:
					if not current_presence_empty:
						test_array[test_array_indices] = np.logical_or(test_array[test_array_indices], current_presence_window)
		return(test_array)
	
	def traverse_embedded_presence_with_demand(self, embedded_presence, local_conv_window_demand):
		start_row_new_data = 0; end_row_new_data = 0; steady_state_new_data = 0
		(_, extra_conv_cols_single_side) = self.convs_min_overlap()
		num_cols = extra_conv_cols_single_side * 2 * self.x_stride + self.filter_cols
		row = self.x_stride; col_count = 0

		for col in range(0, num_cols - self.filter_cols + 1, self.x_stride):
			test_array_indices = tuple([slice(row, row + self.filter_rows), slice(col, col + self.filter_cols), slice(0, self.channels)])
			data_in_current_conv_window = embedded_presence[test_array_indices]
			new_data_count = self.count_new_data(data_in_current_conv_window, local_conv_window_demand) 

			if col < extra_conv_cols_single_side * self.x_stride:
				start_row_new_data += new_data_count
			elif col > self.filter_cols - 1:
				end_row_new_data += new_data_count
			else: 
				steady_state_new_data = new_data_count

			embedded_presence[test_array_indices] = np.logical_or(embedded_presence[test_array_indices], local_conv_window_demand)

		(_, _, _, conv_cols, _) = self.basic_operation_params()
		average_new_data_added = (start_row_new_data + end_row_new_data + steady_state_new_data * (conv_cols - extra_conv_cols_single_side * 2)) / conv_cols
		return(average_new_data_added)
	
	def local_conv_window_basic_movements(self, local_conv_window_demand, conv_idx, first_row, conv_idx_leave_first_row):
		(next_presence_change, current_presence_window) = self.find_spot_in_presence_windows(conv_idx, first_row, conv_idx_leave_first_row)
		embedded_presence_array = self.make_embedded_presence_array(current_presence_window, local_conv_window_demand, first_row)
		return((self.traverse_embedded_presence_with_demand(embedded_presence_array, local_conv_window_demand), next_presence_change))

	def iterate_row_col_fold(self):
		#### ***** still need to add first row extras 
		def calculate_convs_to_fill_SRAM():
			if average_new_data_added == 0:
				return(-1)
			return(effective_SRAM_size / average_new_data_added)

		def manage_conv_target_overreach(conv_target, start_conv_idx):
			nonlocal first_row, effective_SRAM_size, conv_idx
			if conv_target == conv_idx_leave_first_row:
				first_row = 0
				
			remaining_convs = conv_target - start_conv_idx
			remaining_data_reads = remaining_convs * average_new_data_added
			
			self.DRAM_input_reads_analog[self.current_layer] += remaining_data_reads
			conv_idx = conv_target
			effective_SRAM_size -= remaining_data_reads
		
		def manage_full_SRAM():
			nonlocal conv_idx, effective_SRAM_size, conv_idx_last_SRAM_fill, conv_idx_leave_first_row, first_row
			conv_idx += convs_fill_SRAM
			self.DRAM_input_reads_analog[self.current_layer] += effective_SRAM_size
			effective_SRAM_size = self.SRAM_input_size
			conv_idx_last_SRAM_fill = conv_idx
			conv_idx_leave_first_row = min(total_convs, conv_idx + conv_cols)
			first_row = 1
			reset_presence_data()

		def reset_presence_data():
			self.presence_change_indices = [0]; 
			(num_final_rows, _) = self.convs_min_overlap()
			self.presence_change_indices.extend([(conv_rows - row - 1) * conv_cols for row in range(num_final_rows)])
			self.presence_windows = [np.zeros([self.filter_rows, self.filter_cols, self.channels])] * len(self.presence_change_indices)
			self.presence_change_indices.sort(); self.presence_change_indices = np.array(self.presence_change_indices)

		(row_fold, col_fold, conv_rows, conv_cols, total_convs) = self.basic_operation_params()
		effective_SRAM_size = self.SRAM_input_size; 
		print_info = 1
		reset_presence_data()
		
		for col_fold_group in range(col_fold):
			for row_fold_group in range(row_fold):
				local_conv_window_demand = self.make_local_conv_window_demand(row_fold_group)
				conv_idx = 0; first_row = 1; conv_idx_leave_first_row = min(total_convs, conv_cols); conv_idx_last_SRAM_fill = 0
				while (conv_idx < total_convs):
					(average_new_data_added, conv_idx_next_presence_change) = \
						self.local_conv_window_basic_movements(local_conv_window_demand, conv_idx, first_row, conv_idx_leave_first_row)
					
					convs_fill_SRAM = calculate_convs_to_fill_SRAM()
					if (print_info):
						print("")
						print("conv idx: ", conv_idx)
						print("convs to fill SRAM: ", convs_fill_SRAM)
						print("where we will end up if filling sram: ", conv_idx + convs_fill_SRAM)
					input_num = self.input_data_coords(conv_idx)

					if convs_fill_SRAM == -1 or conv_idx + convs_fill_SRAM > conv_idx_next_presence_change:
						manage_conv_target_overreach(conv_idx_next_presence_change, conv_idx)
						if (print_info):
							print("we have overreached target")
							print("new conv idx: ", conv_idx)
							print("new SRAM size: ", effective_SRAM_size)	
							input_num = self.input_data_coords(conv_idx)
				
						if conv_idx_next_presence_change == total_convs: 
							self.add_presence_points(conv_idx_last_SRAM_fill, local_conv_window_demand)
					else: 
						manage_full_SRAM()
						input_num = self.input_data_coords(conv_idx)

	def input_data_coords(self, conv_idx):
		(row_fold, col_fold, conv_rows, conv_cols, total_convs) = self.basic_operation_params()
		conv_row_num = math.floor(conv_idx / conv_cols)
		conv_col_num = conv_idx % conv_cols

		input_row_num = conv_row_num * self.x_stride
		input_col_num = conv_col_num * self.y_stride
		input_num = input_row_num * self.input_cols * self.channels + input_col_num
		return(input_num)
					
	def single_layer_set_params(self, NN_layer):
		self.input_rows  = NN_layer.loc["Input Rows"].item()
		self.input_cols  = NN_layer.loc["Input Columns"].item()
		self.filter_rows = NN_layer.loc["Filter Rows"].item()
		self.filter_cols = NN_layer.loc["Filter Columns"].item()
		self.channels    = NN_layer.loc["Channels"].item()
		self.num_filter  = NN_layer.loc["Num Filter"].item()
		self.x_stride    = NN_layer.loc["X Stride"].item()
		self.y_stride    = NN_layer.loc["Y Stride"].item()

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
			self.add_to_text_output("ERROR. X STRIDE NOT SAME ALL THE WAY ACROSS")
			self.add_to_text_output("Input Cols: " + str(self.input_cols))
			self.add_to_text_output("Better number of input cols: " + str((conv_cols - 1) * self.x_stride + self.filter_cols))

		if ((conv_rows - 1) * self.y_stride + self.filter_rows != self.input_rows):
			self.add_to_text_output("ERROR. Y STRIDE NOT SAME ALL THE WAY ACROSS")
			self.add_to_text_output("Input Rows: " + str(self.input_rows))
			self.add_to_text_output("Better number of input rows: " + str((conv_rows - 1) * self.y_stride + self.filter_rows))

		self.num_compute_clock_cycles_analog[self.current_layer]  = self.batch_size * num_conv_in_input * col_fold * row_fold
		self.num_compute_clock_cycles_digital[self.current_layer] = -1
		self.num_program_compute_instance[self.current_layer]     = row_fold * col_fold
		self.num_program_clock_cycles[self.current_layer]         = -1

		self.SRAM_input_reads[self.current_layer]      = self.batch_size * num_conv_in_input * ind_filter_size * col_fold
		self.SRAM_filter_reads[self.current_layer]     = ind_filter_size * self.num_filter
		self.SRAM_output_writes_SS[self.current_layer] = self.batch_size * num_conv_in_input *  self.num_filter * row_fold 

		self.DRAM_filter_reads[self.current_layer] = ind_filter_size * self.num_filter
		self.DRAM_output_writes_SS[self.current_layer] = self.SRAM_output_writes_SS[self.current_layer]
		# not doing input here b/c that's what all the fancy pseudo-analytical sim tools are for
		
		self.SRAM_output_writes_acc[self.current_layer] = -1
		self.SRAM_output_reads_acc[self.current_layer] = -1
		self.DRAM_output_reads_acc[self.current_layer] = -1
		self.DRAM_output_writes_acc[self.current_layer] = -1
		self.DRAM_input_reads_digital_SRAM_sharing[self.current_layer] = -1
		self.DRAM_output_writes_acc_SRAM_sharing[self.current_layer] = -1
	
		self.compute_input_DRAM_access()

	def calculate_NN_totals(self):
		self.num_compute_clock_cycles_analog_total  = sum(self.num_compute_clock_cycles_analog)
		self.num_compute_clock_cycles_digital_total = sum(self.num_compute_clock_cycles_digital)
		self.num_program_compute_instance_total     = sum(self.num_program_compute_instance)
		self.num_program_clock_cycles_total         = sum(self.num_program_clock_cycles)

		self.SRAM_input_reads_total      = sum(self.SRAM_input_reads)
		self.SRAM_filter_reads_total     = sum(self.SRAM_filter_reads)
		self.SRAM_output_writes_SS_total = sum(self.SRAM_output_writes_SS)

		self.DRAM_filter_reads_total     = sum(self.DRAM_filter_reads)
		self.DRAM_output_writes_SS_total = sum(self.DRAM_output_writes_SS)

		self.DRAM_input_reads_analog_total  = sum(self.DRAM_input_reads_analog)
		self.DRAM_input_reads_digital_total = sum(self.DRAM_input_reads_digital)

		self.SRAM_output_writes_acc_total = sum(self.SRAM_output_writes_acc)
		self.SRAM_output_reads_acc_total  = sum(self.SRAM_output_reads_acc)
		self.DRAM_output_reads_acc_total  = sum(self.DRAM_output_reads_acc)
		self.DRAM_output_writes_acc_total = sum(self.DRAM_output_writes_acc)

		self.DRAM_input_reads_digital_SRAM_sharing_total   = sum(self.DRAM_input_reads_digital_SRAM_sharing)
		self.DRAM_output_writes_acc_SRAM_sharing_total     = sum(self.DRAM_output_writes_acc_SRAM_sharing)

	def return_specs(self):
		runspecs_names = ["SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
			"DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes", \
			"Total Program/Compute Instances", "Total Programming Clock Cycles", \
			"Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital"]
	
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_SS_total, \
			self.DRAM_input_reads_digital_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_SS_total, \
			self.num_program_compute_instance_total, -1, \
			self.num_compute_clock_cycles_analog_total, -1]


		return(pd.DataFrame(totals, runspecs_names))
