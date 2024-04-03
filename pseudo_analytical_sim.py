import math
import time
#import SRAM_model
import pandas as pd
import numpy as np



class hardware_state():
	def __init__(self, compute_type):
		self.compute_type = compute_type

	def add_to_text_output(self, string_output):
		self.text_output += "\n" + string_output
		x = 1

	def set_hardware(self, hardware_state_info):
		self.array_rows = hardware_state_info.loc["Systolic Array Rows"]#.item()
		self.array_cols = hardware_state_info.loc["Systolic Array Cols"]#.item()
		self.SRAM_input_size = hardware_state_info.loc["SRAM Input Size"] * 1024 / 2 #.item() * 1000 / 2
		self.SRAM_filter_size = hardware_state_info.loc["SRAM Filter Size"] * 1024 / 2 #.item() * 1000 / 2
		self.SRAM_output_size = hardware_state_info.loc["SRAM Output Size"] * 1024 / 2 #.item() * 1000 / 2
		self.batch_size = hardware_state_info.loc["Batch Size"]#.item()
			
		#self.input_SRAM  = SRAM_model.SRAM_model(self.SRAM_input_size, "input")
		#self.filter_SRAM = SRAM_model.SRAM_model(self.SRAM_filter_size, "filter")
		#self.output_SRAM = SRAM_model.SRAM_model(self.SRAM_output_size, "output")

		self.text_output = ""

	def set_NN(self, NN_layers_all):
		self.NN_layers_all = NN_layers_all
		self.num_NN_layers = len(NN_layers_all)

	def set_results_vars(self):
		self.num_program_compute_instance     = [0] * self.num_NN_layers
		#self.num_compute_clock_cycles_analog  = [0] * self.num_NN_layers
		#self.num_compute_clock_cycles_digital = [0] * self.num_NN_layers
		self.num_compute_clock_cycles = [0] * self.num_NN_layers

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
		AM_execution_time = (end_time - start_time) / 60
		self.calculate_NN_totals()
		final_time = time.time()
		AM_post_process_time = (final_time - end_time) / 60,
				
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
		stride_factor_rows = min(self.filter_rows / self.x_stride, 1)
		stride_factor_cols = min(self.filter_cols / self.y_stride, 1)
		input_size = self.input_rows * self.input_cols * self.channels * stride_factor_cols * stride_factor_rows

		if 0:#(self.SRAM_input_size >= input_size):
			self.DRAM_input_reads_analog[self.current_layer] = input_size
			self.DRAM_input_reads_analog = np.array(self.DRAM_input_reads_analog)
			self.add_to_text_output("SRAM can fit entirety of input data")
		else:
			self.add_to_text_output("SRAM canNOT fit entirety of input data")
			self.iterate_row_col_fold()

	def make_repeat_access_list(self):
		col_patterns = min(self.filter_cols % self.x_stride + 1, self.filter_cols)
		row_patterns = min(self.filter_rows % self.y_stride + 1, self.filter_rows)
		repeat_access_list_diff = [] # * (col_patterns * row_patterns)

		pixel_base_cc_all = np.ones([row_patterns, col_patterns, self.filter_rows, self.filter_cols]) * -1
		row_fold_group_base_cc_all = np.ones([row_patterns, col_patterns, self.filter_rows, self.filter_cols]) * -1
		flattened_filter_index_eff_all = np.ones([row_patterns, col_patterns, self.filter_rows, self.filter_cols]) * -1
		result = np.ones([row_patterns, col_patterns, self.filter_rows, self.filter_cols]) * -1


		channel = 0
		data_per_row = self.filter_cols * self.channels
		data_per_pixel = self.channels

		for row_pattern in range(row_patterns):
			for col_pattern in range(col_patterns):
				single_stride_repeat_access_list = []
				for row in range(row_pattern, self.filter_rows, self.y_stride):
					for col in range(col_pattern, self.filter_cols, self.x_stride):
						if (col == 0) and (row == 2):
							x = 1
						#print(row, col, "\n")
						flattened_filter_index = row * data_per_row + data_per_pixel * col + channel
						row_fold_group = math.floor(flattened_filter_index / self.array_rows)
						row_fold_group_base_cc = row_fold_group * self.total_convs
						#print(row_fold_group_base_cc)

						flattened_filter_index_eff = flattened_filter_index % self.array_rows
						pixel_base_cc = -(self.conv_cols * (row / self.y_stride) + (col / self.x_stride))
						## this pixel base cc is what would have to change but it doesn't b/c already 
						## accounts for stride
						#print(pixel_base_cc)
						val = row_fold_group_base_cc + pixel_base_cc

						if self.compute_type == "digital":
							val += flattened_filter_index_eff
						
						for col_fold_group in range(self.col_fold):
							single_stride_repeat_access_list.append(val + col_fold_group * self.total_convs * self.row_fold)
						
						pixel_base_cc_all[row_pattern, col_pattern, row, col] = pixel_base_cc
						row_fold_group_base_cc_all[row_pattern, col_pattern, row, col] = row_fold_group_base_cc
						flattened_filter_index_eff_all[row_pattern, col_pattern, row, col] = flattened_filter_index_eff
						result[row_pattern, col_pattern, row, col] = val


				single_stride_repeat_access_list.sort()

				single_stride_repeat_access_list_diff = [single_stride_repeat_access_list[i+1] - single_stride_repeat_access_list[i] for i in range(len(single_stride_repeat_access_list) - 1)]
				single_stride_repeat_access_list_diff = np.append(single_stride_repeat_access_list_diff, [np.Infinity])
				repeat_access_list_diff.extend(single_stride_repeat_access_list_diff)
		
		repeat_access_list_diff.sort()
		approximate_cc_fill = self.SRAM_input_size / ((self.input_cols * self.input_rows * self.channels) /  self.total_convs)
		#first_revolution = min
		#repeat_access_list_diff = [x for x in repeat_access_list_diff if x < approximate_cc_fill]
		self.repeat_access_list_diff = repeat_access_list_diff


	def traverse_repeat_access_array(self):

		#self.SRAM_input_size = 2.1 * 1024
		alignment_factor = 0.5
		local_base_conv_idx = 0; absolute_conv_idx = 0
		self.SRAM_free_space = self.SRAM_input_size
		repeat_access_list_diff = np.array(self.repeat_access_list_diff)

		#for col_fold_group in range(self.col_fold):
		#	for row_fold_group in range(self.row_fold):
		#x = np.array([127, 540])
		#repeat_access_list_diff = x
		#repeat_access_list_diff = np.array([127, 527])
		while(absolute_conv_idx < self.total_convs * self.row_fold * self.col_fold):
			relative_conv_idx = absolute_conv_idx - local_base_conv_idx

			eff_SA_rows = self.ind_filter_size / self.row_fold
			new_data_per_conv = sum(repeat_access_list_diff > relative_conv_idx + alignment_factor) * eff_SA_rows / len(repeat_access_list_diff)
			#new_data_per_conv_x = sum(x > relative_conv_idx) * eff_SA_rows / len(x)

			if relative_conv_idx >= max(repeat_access_list_diff):
				return

			next_change_repeat_access = min(repeat_access_list_diff[repeat_access_list_diff > relative_conv_idx + alignment_factor]) 
			next_change_repeat_access -= relative_conv_idx
			#print(new_data_per_conv)
			
			next_change_repeat_access = min(self.total_convs * self.row_fold * self.col_fold - absolute_conv_idx, next_change_repeat_access)
			#convs_change_repeat_access = next_change_repeat_access - relative_conv_idx
			convs_change_repeat_access = next_change_repeat_access
			convs_fill_SRAM = self.SRAM_free_space / new_data_per_conv

			if convs_change_repeat_access < convs_fill_SRAM:
				self.DRAM_input_reads_analog += new_data_per_conv * convs_change_repeat_access
				self.SRAM_free_space -= new_data_per_conv * convs_change_repeat_access
				absolute_conv_idx += convs_change_repeat_access
				#print(convs_change_repeat_access)

			elif convs_fill_SRAM <= convs_change_repeat_access:
				self.SRAM_unfilled = 0
				self.DRAM_input_reads_analog = self.SRAM_free_space + self.DRAM_input_reads_analog
				self.SRAM_free_space = self.SRAM_input_size
				absolute_conv_idx += convs_fill_SRAM
				local_base_conv_idx = absolute_conv_idx
				#print(convs_fill_SRAM)
			#print()
			#print()

	def iterate_row_col_fold(self):
		self.make_repeat_access_list()
		self.traverse_repeat_access_array()
	
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

		#self.num_compute_clock_cycles_analog[self.current_layer]  = self.batch_size * self.num_conv_in_input * self.col_fold * self.row_fold
		#self.num_compute_clock_cycles_digital[self.current_layer] = self.num_compute_clock_cycles_analog[self.current_layer] + (self.ind_filter_size - 1) % self.array_rows
		#self.num_compute_clock_cycles_digital[self.current_layer] = self.batch_size * ((self.ind_filter_size - 1) % self.array_rows + self.num_conv_in_input) * self.row_fold * self.col_fold 

		row_rem = self.ind_filter_size % self.array_rows
		col_rem = self.num_filter % self.array_cols

		x = self.batch_size * self.num_conv_in_input * self.col_fold * self.row_fold
		if self.compute_type == "analog":
			self.num_compute_clock_cycles[self.current_layer] = x
		else:
			x += (self.array_rows - 1 + self.array_cols - 1) * (self.row_fold - 1) * (self.col_fold - 1)
			x += (row_rem + self.array_cols) * (self.col_fold - 1)
			x += (self.array_rows - 1 + col_rem) * (self.row_fold - 1)
			x += row_rem + col_rem
			self.num_compute_clock_cycles[self.current_layer] = x



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
		self.DRAM_input_reads_digital = self.DRAM_input_reads_analog.astype(int)

	def calculate_NN_totals(self):
		#self.num_compute_clock_cycles_analog_total  = sum(self.num_compute_clock_cycles_analog)
		#self.num_compute_clock_cycles_digital_total = sum(self.num_compute_clock_cycles_digital)
		self.num_compute_clock_cycles_total = sum(self.num_compute_clock_cycles)

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
			"Total Compute Clock Cycles"]
	
		totals = [self.SRAM_input_reads_total, self.SRAM_filter_reads_total, self.SRAM_output_writes_SS_total, \
			self.DRAM_input_reads_digital_total, self.DRAM_filter_reads_total, self.DRAM_output_writes_SS_total, \
			self.num_program_compute_instance_total, -1, \
			self.num_compute_clock_cycles_total]

		return(pd.DataFrame(totals, runspecs_names))
	
	def return_specs_self_compare(self):
		# need to show effects of 1) accumulator modeling, 2) SRAM sharing
		# 3) digital vs analog clock cycles, 4) digital vs analog DRAM input reads
		runspecs_names = ["Total Compute Clock Cycles",\
			"DRAM Input Reads Analog", "DRAM Input Reads Digital", \
			"SRAM Output Writes SS", "DRAM Output Writes SS",\
			"SRAM Output Writes Acc", "SRAM Output Reads Acc",\
			"DRAM Output Writes Acc", "DRAM Output Reads Acc", \
			"DRAM Input Reads Digital SRAM Sharing", "DRAM Output Writes Digital Acc SRAM Sharing"]
		
		totals = [self.num_compute_clock_cycles_total,\
		self.DRAM_input_reads_analog_total, self.DRAM_input_reads_digital_total,\
		self.SRAM_output_writes_SS_total, self.DRAM_output_writes_SS_total, \
		self.SRAM_output_writes_acc_total, self.SRAM_output_reads_acc_total, \
		self.DRAM_output_writes_acc_total, self.DRAM_output_reads_acc_total, \
		self.DRAM_input_reads_digital_SRAM_sharing_total, self.DRAM_output_writes_acc_SRAM_sharing_total]

		return(pd.DataFrame(totals, runspecs_names))