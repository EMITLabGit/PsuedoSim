import math
import time
import SRAM_model
import pandas as pd
import numpy as np

class hardware_state():
	def __init__(self, compute_type):
		self.compute_type = compute_type

	def add_to_text_output(self, string_output):
		self.text_output += "\n" + string_output

	def set_hardware(self, hardware_state_info):
		self.array_rows = hardware_state_info.loc["Systolic Array Rows"]#.item()
		self.array_cols = hardware_state_info.loc["Systolic Array Cols"]#.item()
		self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"] * 1024 / 2 #.item() * 1000 / 2
		self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"] * 1024 / 2 #.item() * 1000 / 2
		self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"] * 1024 / 2 #.item() * 1000 / 2
		self.batch_size = hardware_state_info.loc["Batch Size"]#.item()
			
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

		self.SRAM_output_reads_acc  = [0] * self.num_NN_layers
		self.SRAM_output_writes_acc = [0] * self.num_NN_layers
		self.DRAM_output_reads_acc  = [0] * self.num_NN_layers
		self.DRAM_output_writes_acc = [0] * self.num_NN_layers

		self.DRAM_input_reads_digital_SRAM_sharing = [0] * self.num_NN_layers
		self.DRAM_output_writes_acc_SRAM_sharing   = [0] * self.num_NN_layers

	def run_single_layer(self, NNLayer, hardwareArch):
		self.num_NN_layers = 1
		self.current_layer = 0
		self.set_results_vars()
		start_time = time.time()

		self.set_hardware(hardwareArch)			
		self.single_layer_set_params(NNLayer)

		end_time = time.time()
		AM_execution_time = round((end_time - start_time) / 60, 5)
		self.calculate_NN_totals()
		final_time = time.time()
		AM_post_process_time = round((final_time - end_time) / 60, 5)
				
		AM_results_SS_compare = self.return_specs_SS_compare()
		AM_results_self_compare = self.return_specs_self_compare()
		AM_results_SS_compare.loc[" "] = " "
		AM_results_SS_compare.loc["Simulation Run Time [min]"] = AM_execution_time
		AM_results_SS_compare.loc["Simulation Post Process Time [min]"] = AM_post_process_time

		return(AM_results_SS_compare, AM_results_self_compare, self.text_output)

	def run_all_layers(self):
		self.set_results_vars()
		start_time = time.time()
		for index, layer in enumerate(self.NN_layers_all):
			self.current_layer = index
			self.SRAM_carryover_data_current_layer = 0; self.SRAM_carryover_data_previous_layer = 0
			status = self.single_layer_set_params(layer)
			if (status == -1):
				return
		end_time = time.time()
		AM_execution_time = round((end_time - start_time) / 60, 10)
		self.calculate_NN_totals()
		final_time = time.time()
		AM_post_process_time = round((final_time - end_time) / 60, 10)
		AM_results_SS_compare = self.return_specs_SS_compare()
		AM_results_self_compare = self.return_specs_self_compare()
		AM_results_SS_compare.loc[" "] = " "
		AM_results_SS_compare.loc["Simulation Run Time [min]"] = AM_execution_time
		AM_results_SS_compare.loc["Simulation Post Process Time [min]"] = AM_post_process_time
		return(AM_results_SS_compare, AM_results_self_compare, self.text_output)

	def compute_input_DRAM_access(self):
		input_size = self.input_rows * self.input_cols * self.channels
		if (self.SRAM_input_size >= input_size):
			self.DRAM_input_reads_analog[self.current_layer] = input_size
			self.add_to_text_output("SRAM can fit entirety of input data")
		else:
			self.add_to_text_output("SRAM canNOT fit entirety of input data")
			self.iterate_row_col_fold()
			#self.DRAM_input_reads_analog[self.current_layer] = int(self.DRAM_input_reads_analog[self.current_layer])

	def make_global_repeat_access_matrix(self):
		col_patterns = self.filter_cols % self.x_stride + 1
		row_patterns = self.filter_rows % self.y_stride + 1
		repeat_access_matrix = np.ones([row_patterns, col_patterns, self.filter_rows, self.filter_cols]) * np.NAN
		channel = 0
		data_per_row = self.filter_cols * self.channels
		for row_pattern in range(row_patterns):
			for col_pattern in range(col_patterns):
				for row in range(0, self.filter_rows, self.y_stride):
					for col in range(0, self.filter_cols, self.x_stride):
						if (row_pattern, col_pattern) == (0, 0):
							flattened_filter_index = row * data_per_row + self.channels * col + channel
							row_fold_group = math.floor(flattened_filter_index / self.array_rows)
							row_fold_group_base_cc = row_fold_group * self.total_convs

							flattened_filter_index_eff = flattened_filter_index % self.array_rows
							row_eff = math.floor(flattened_filter_index_eff / data_per_row)
							col_eff = math.floor((flattened_filter_index_eff % data_per_row) / self.channels)
							#pixel_base_cc = -(self.conv_cols * (row_eff / self.y_stride) + (col_eff / self.x_stride))
							pixel_base_cc = -(self.conv_cols * (row / self.y_stride) + (col / self.x_stride))

							repeat_access_matrix[row_pattern, col_pattern, row, col] = row_fold_group_base_cc + pixel_base_cc

							if self.compute_type == "digital":
								repeat_access_matrix[row_pattern, col_pattern, row, col] += flattened_filter_index_eff

						else: 
							# this stays the same
							repeat_access_matrix[row_pattern, col_pattern, row_pattern:, col_pattern:] = repeat_access_matrix[0, 0, 0:self.filter_rows - row_pattern, 0:self.filter_cols - col_pattern]
				
				repeat_access_matrix[row_pattern, col_pattern, :, :] = repeat_access_matrix[row_pattern, col_pattern, :, :] - np.nanmin(repeat_access_matrix[row_pattern, col_pattern, :, :])
				
		#repeat_access_matrix = repeat_access_matrix - np.min(repeat_access_matrix)
		self.repeat_access_matrix = repeat_access_matrix

	def make_local_repeat_access_matrix(self, flat_filter_access_start_idx_no_channel, \
				     flat_filter_access_end_idx_no_channel):
		#local_repeat_access_matrix = np.ones([self.filter_rows, self.filter_cols]) * -1
		
		if flat_filter_access_end_idx_no_channel == self.filter_rows * self.filter_cols:
			flat_filter_access_end_idx_no_channel -= 1
		flat_filter_access_range_no_channel = np.arange(flat_filter_access_start_idx_no_channel, \
						  flat_filter_access_end_idx_no_channel + 1)
		(row_ind, col_ind) = np.unravel_index(flat_filter_access_range_no_channel, [self.filter_rows, self.filter_cols])

		local_repeat_access_all_stride = []
		for row_pattern in range(self.repeat_access_matrix.shape[0]):
			for col_pattern in range(self.repeat_access_matrix.shape[1]):
				local_repeat_access_single_stride = self.repeat_access_matrix[row_pattern, col_pattern, row_ind, col_ind]	
				if not np.array_equal(local_repeat_access_single_stride, np.array([np.NAN] * len(local_repeat_access_single_stride)), equal_nan = True):
					local_repeat_access_single_stride.sort()
					local_repeat_access_single_stride_diff = [local_repeat_access_single_stride[i+1] - local_repeat_access_single_stride[i] for i in range(len(local_repeat_access_single_stride) - 1)]
					local_repeat_access_single_stride_diff = np.append(local_repeat_access_single_stride_diff, [np.Infinity])
					local_repeat_access_single_stride_diff.sort()
					local_repeat_access_all_stride.extend(local_repeat_access_single_stride_diff[np.invert(np.isnan(local_repeat_access_single_stride_diff))])


		'''
		local_repeat_access_diff = np.zeros(local_repeat_access.shape)
		for idx in range(local_repeat_access_diff.shape[0]):
			if local_repeat_access[idx] == np.max(local_repeat_access):
				local_repeat_access_diff[idx] = np.Infinity
			else:
				local_repeat_access_temp = local_repeat_access.copy()
				local_repeat_access_temp[idx] = np.max(local_repeat_access_temp) + 1
				local_repeat_access_temp[local_repeat_access_temp < local_repeat_access[idx]] \
					= np.max(local_repeat_access_temp) + 1
				local_repeat_access_diff[idx] = np.min(np.abs(local_repeat_access_temp - local_repeat_access[idx]))

		local_repeat_access_diff.sort()
		'''
		local_repeat_access_all_stride.sort()
		local_repeat_access_all_stride = np.array(local_repeat_access_all_stride)
		return(local_repeat_access_all_stride)

	def traverse_repeat_access_arrays(self, local_repeat_access_start_channels, local_repeat_access_mid_channels,\
				    local_repeat_access_end_channels, num_start_channels, num_mid_channels, num_end_channels):
		
		alignment_factor = 0.5
		local_base_conv_idx = 0; absolute_conv_idx = 0
		#self.SRAM_free_space = self.SRAM_input_size
		while(absolute_conv_idx < self.total_convs):
			relative_conv_idx = absolute_conv_idx - local_base_conv_idx

			start_channels_new_data_per_conv = sum(local_repeat_access_start_channels > relative_conv_idx) * num_start_channels
			mid_channels_new_data_per_conv   = sum(local_repeat_access_mid_channels   > relative_conv_idx) * num_mid_channels
			end_channels_new_data_per_conv   = sum(local_repeat_access_end_channels   > relative_conv_idx) * num_end_channels

			next_change_repeat_access_start = np.Inf
			next_change_repeat_access_mid   = np.Inf
			next_change_repeat_access_end   = np.Inf

			if local_repeat_access_start_channels.size != 0: 
				next_change_repeat_access_start = min(local_repeat_access_start_channels[local_repeat_access_start_channels > relative_conv_idx + alignment_factor]) 
			if local_repeat_access_mid_channels.size != 0:
				next_change_repeat_access_mid   = min(local_repeat_access_mid_channels[local_repeat_access_mid_channels   > relative_conv_idx + alignment_factor])
			if local_repeat_access_end_channels.size != 0:
				next_change_repeat_access_end = min(local_repeat_access_end_channels[local_repeat_access_end_channels   > relative_conv_idx + alignment_factor])

			convs_change_repeat_access = min(next_change_repeat_access_start, min(next_change_repeat_access_mid, next_change_repeat_access_end)) - relative_conv_idx
			convs_change_repeat_access = min(self.total_convs - absolute_conv_idx, convs_change_repeat_access)

			if convs_change_repeat_access == np.Infinity:
				convs_change_repeat_access = self.total_convs - absolute_conv_idx
				
			new_data_per_conv = (start_channels_new_data_per_conv + mid_channels_new_data_per_conv + end_channels_new_data_per_conv)
			convs_fill_SRAM = self.SRAM_free_space / new_data_per_conv

			if convs_change_repeat_access < convs_fill_SRAM:
				self.DRAM_input_reads_analog += new_data_per_conv * convs_change_repeat_access
				self.SRAM_free_space -= new_data_per_conv * convs_change_repeat_access
				absolute_conv_idx += convs_change_repeat_access
				#print("\nSRAM fill")

			elif convs_fill_SRAM < convs_change_repeat_access:
				self.SRAM_unfilled = 0
				self.DRAM_input_reads_analog = self.SRAM_free_space + self.DRAM_input_reads_analog
				self.SRAM_free_space = self.SRAM_input_size
				absolute_conv_idx += convs_fill_SRAM
				local_base_conv_idx = absolute_conv_idx
				#print("\nconv repeat change")
			
			if (0):
				print("end of loop")
				print("absolute conv index", absolute_conv_idx)
				print("relative conv index", relative_conv_idx)
				print("new data per conv", new_data_per_conv)
				print("convs fill SRAM", convs_fill_SRAM)
				print("convs change repeat access", convs_change_repeat_access)


	def make_local_conv_window_demand(self, row_fold_group):
		flat_filter_access_start_idx = row_fold_group * self.array_rows
		flat_filter_access_end_idx = min(self.filter_rows * self.filter_cols * self.channels, (row_fold_group + 1) * self.array_rows) - 1 # i think the -1 is right?

		flat_filter_access_start_idx_no_channel = math.floor(flat_filter_access_start_idx / self.channels)
		flat_filter_access_end_idx_no_channel = math.floor(flat_filter_access_end_idx / self.channels)

		(_, _, channel_ind) = np.unravel_index([flat_filter_access_start_idx, flat_filter_access_end_idx], [self.filter_rows, self.filter_cols, self.channels])
		start_pixel_channel = channel_ind[0]; end_pixel_channel = channel_ind[1]

		if start_pixel_channel > end_pixel_channel:
			# low channel - end only - this region ends at end pixel channel
			local_repeat_access_start_channels = self.make_local_repeat_access_matrix(flat_filter_access_start_idx_no_channel + 1, \
					flat_filter_access_end_idx_no_channel)
			# medium channel - neither
			local_repeat_access_mid_channels = self.make_local_repeat_access_matrix(flat_filter_access_start_idx_no_channel + 1, \
					flat_filter_access_end_idx_no_channel - 1)
			# high channel - start only - this region starts at start pixel channel
			local_repeat_access_end_channels = self.make_local_repeat_access_matrix(flat_filter_access_start_idx_no_channel, \
					flat_filter_access_end_idx_no_channel - 1)
			
			self.traverse_repeat_access_arrays(local_repeat_access_start_channels, local_repeat_access_mid_channels, \
				      local_repeat_access_end_channels, end_pixel_channel, start_pixel_channel - end_pixel_channel + 1, \
						self.channels - start_pixel_channel - 1)

		else: 
			# low channel - end only - this region ends at start pixel channel 
			local_repeat_access_start_channels = self.make_local_repeat_access_matrix(\
				flat_filter_access_start_idx_no_channel + 1, flat_filter_access_end_idx_no_channel)
			# medium channel - both 
			local_repeat_access_mid_channels = self.make_local_repeat_access_matrix(\
				flat_filter_access_start_idx_no_channel, flat_filter_access_end_idx_no_channel)
			# high channel - start only - this region starts at end pixel channel
			local_repeat_access_end_channels = self.make_local_repeat_access_matrix(\
				flat_filter_access_start_idx_no_channel, flat_filter_access_end_idx_no_channel - 1)

			self.traverse_repeat_access_arrays(local_repeat_access_start_channels, local_repeat_access_mid_channels, local_repeat_access_end_channels, \
				       start_pixel_channel, end_pixel_channel - start_pixel_channel + 1, self.channels - end_pixel_channel - 1)

	def reset_presence_data(self):
		self.presence_change_indices = [0]; 
		num_final_rows = self.convs_min_overlap_x
		self.presence_change_indices.extend([(self.conv_rows - row - 1) * self.conv_cols for row in range(num_final_rows)])
		self.presence_windows = [np.zeros([self.filter_rows, self.filter_cols, self.channels])] * len(self.presence_change_indices)
		self.presence_change_indices.sort(); 
		self.presence_change_indices = np.array(self.presence_change_indices)

	def iterate_row_col_fold(self):
		effective_SRAM_size = self.SRAM_input_size; 
		self.reset_presence_data()
		self.make_global_repeat_access_matrix()
		#print(self.repeat_access_matrix)
		#for col_fold_group in range(self.col_fold):
		self.SRAM_unfilled = 1
		self.SRAM_free_space = self.SRAM_input_size
		for row_fold_group in range(self.row_fold):
			self.make_local_conv_window_demand(row_fold_group)
		if not self.SRAM_unfilled:
			self.DRAM_input_reads_analog *= self.col_fold
	
	def basic_operation_params(self):
		self.ind_filter_size = self.filter_rows * self.filter_cols * self.channels
		self.row_fold = math.ceil(self.ind_filter_size / self.array_rows)
		self.col_fold = math.ceil(self.num_filter / self.array_cols)  
		self.conv_rows = math.ceil((self.input_rows - self.filter_rows) / self.x_stride) + 1 # math.ceil(self.input_rows / stride)
		self.conv_cols = math.ceil((self.input_cols - self.filter_cols) / self.y_stride) + 1 # math.ceil(self.input_cols / stride)

		self.conv_rows = math.ceil((self.input_rows - self.filter_rows + self.x_stride) / self.x_stride) 
		self.conv_cols = math.ceil((self.input_cols - self.filter_cols + self.y_stride) / self.y_stride) 

		while(self.conv_rows * self.x_stride > self.input_rows):
			self.conv_rows -= 1
		while(self.conv_cols * self.y_stride > self.input_cols):
			self.conv_cols -= 1
			
		self.num_conv_in_input = self.input_cols * self.input_rows
		self.num_conv_in_input = self.conv_cols * self.conv_rows
		self.convs_min_overlap_x = math.floor((self.filter_rows - 1) / self.y_stride)
		self.convs_min_overlap_y = math.floor((self.filter_cols - 1) / self.x_stride)

		self.total_convs = self.conv_cols * self.conv_rows

	def set_NN_layer(self, NN_layer):
		self.input_rows  = NN_layer.loc["Input Rows"].item()
		self.input_cols  = NN_layer.loc["Input Columns"].item()
		self.filter_rows = NN_layer.loc["Filter Rows"].item()
		self.filter_cols = NN_layer.loc["Filter Columns"].item()
		self.channels    = NN_layer.loc["Channels"].item()
		self.num_filter  = NN_layer.loc["Num Filter"].item()
		self.x_stride    = NN_layer.loc["X Stride"].item()
		self.y_stride    = NN_layer.loc["Y Stride"].item()		

	def single_layer_set_params(self, NN_layer):
		self.set_NN_layer(NN_layer)
		self.basic_operation_params()

		self.num_compute_clock_cycles_analog[self.current_layer]  = self.batch_size * self.num_conv_in_input * self.col_fold * self.row_fold
		self.num_compute_clock_cycles_digital[self.current_layer] = self.num_compute_clock_cycles_analog[self.current_layer] + (self.ind_filter_size - 1) % self.array_rows
		self.num_compute_clock_cycles_digital[self.current_layer] = self.batch_size * ((self.ind_filter_size - 1) % self.array_rows + self.num_conv_in_input) * self.row_fold * self.col_fold 

		self.num_program_compute_instance[self.current_layer]     = self.row_fold * self.col_fold
		self.num_program_clock_cycles[self.current_layer]         = -1

		self.SRAM_input_reads[self.current_layer]      = self.batch_size * self.num_conv_in_input * self.ind_filter_size * self.col_fold
		self.SRAM_filter_reads[self.current_layer]     = self.ind_filter_size * self.num_filter
		self.SRAM_output_writes_SS[self.current_layer] = self.batch_size * self.num_conv_in_input *  self.num_filter * self.row_fold 

		self.DRAM_filter_reads[self.current_layer] = self.ind_filter_size * self.num_filter
		self.DRAM_output_writes_SS[self.current_layer] = self.SRAM_output_writes_SS[self.current_layer]
		
		self.SRAM_output_reads_acc[self.current_layer] = -1
		self.SRAM_output_writes_acc[self.current_layer] = -1
		self.DRAM_output_reads_acc[self.current_layer] = -1
		self.DRAM_output_writes_acc[self.current_layer] = -1
		self.DRAM_input_reads_digital_SRAM_sharing[self.current_layer] = -1
		self.DRAM_output_writes_acc_SRAM_sharing[self.current_layer] = -1
	
		self.compute_input_DRAM_access()
		self.DRAM_input_reads_digital = [-1]#self.DRAM_input_reads_analog.astype(int)

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

		self.SRAM_output_reads_acc_total  = sum(self.SRAM_output_reads_acc)
		self.SRAM_output_writes_acc_total = sum(self.SRAM_output_writes_acc)
		self.DRAM_output_reads_acc_total  = sum(self.DRAM_output_reads_acc)
		self.DRAM_output_writes_acc_total = sum(self.DRAM_output_writes_acc)

		self.DRAM_input_reads_digital_SRAM_sharing_total   = sum(self.DRAM_input_reads_digital_SRAM_sharing)
		self.DRAM_output_writes_acc_SRAM_sharing_total     = sum(self.DRAM_output_writes_acc_SRAM_sharing)

	def return_specs_SS_compare(self):
		runspecs_names = ["SRAM Input Reads", "SRAM Filter Reads", "SRAM Output Writes", \
			"DRAM Input Reads", "DRAM Filter Reads", "DRAM Output Writes", \
			"Total Program/Compute Instances", "Total Programming Clock Cycles", \
			"Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital"]
	
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_SS_total, \
			self.DRAM_input_reads_digital_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_SS_total, \
			self.num_program_compute_instance_total, -1, \
			self.num_compute_clock_cycles_analog_total, self.num_compute_clock_cycles_digital_total]

		return(pd.DataFrame(totals, runspecs_names))
	
	def return_specs_self_compare(self):
		# need to show effects of 1) accumulator modeling, 2) SRAM sharing
		# 3) digital vs analog clock cycles, 4) digital vs analog DRAM input reads
		runspecs_names = ["Total Compute Clock Cycles Analog", "Total Compute Clock Cycles Digital", \
			"DRAM Input Reads Analog", "DRAM Input Reads Digital", \
			"SRAM Output Writes SS", "DRAM Output Writes SS",\
			"SRAM Output Writes Acc", "SRAM Output Reads Acc",\
			"DRAM Output Writes Acc", "DRAM Output Reads Acc", \
			"DRAM Input Reads Digital SRAM Sharing", "DRAM Output Writes Digital Acc SRAM Sharing"]
		
		totals = [self.num_compute_clock_cycles_analog_total, self.num_compute_clock_cycles_digital_total, \
		self.DRAM_input_reads_analog_total, self.DRAM_input_reads_digital_total,\
		self.SRAM_output_writes_SS_total, self.DRAM_output_writes_SS_total, \
		self.SRAM_output_writes_acc_total, self.SRAM_output_reads_acc_total, \
		self.DRAM_output_writes_acc_total, self.DRAM_output_reads_acc_total, \
		self.DRAM_input_reads_digital_SRAM_sharing_total, self.DRAM_output_writes_acc_SRAM_sharing_total]

		return(pd.DataFrame(totals, runspecs_names))