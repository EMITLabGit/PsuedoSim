import math

class SRAM_model():
     def __init__(self, memory_size_words, name):
          self.full = 0
          self.fill_size = 0
          self.DRAM_reads = []
          self.component_misses = []
          self.component_accesses = []
          self.memory_size_words = memory_size_words
          self.name = name
          
     def new_layer(self, component_size_words, num_component, carryover_num_words = 0):
          if (component_size_words > self.memory_size_words):
               print("SIMULATION CANNOT BE RUN B/C ONE OF THE SRAMs IS NOT LARGE ENOUGH TO HOLD A SINGLE COMPONENT")
               print("Component Size is: ", component_size_words, "Memory size is: ", self.memory_size_words)
               return (-1)

          self.component_size_words = component_size_words
          self.num_component = num_component
          self.memory_size_num_component = math.floor(self.memory_size_words / component_size_words)
          self.component_status = [0] * num_component

          if carryover_num_words > self.memory_size_words:
               carryover_num_words = self.memory_size_words          
          carryover_num_components = math.floor(carryover_num_words / component_size_words)
          if carryover_num_components > num_component:
               carryover_num_components = num_component
          self.component_status[0:carryover_num_components] = [1] * carryover_num_components
          self.fill_size = carryover_num_components # test this
          if self.fill_size == self.memory_size_num_component: # this
               self.full = 1 # this

          self.DRAM_reads_layer = 0
          self.component_misses_layer = 0
          self.component_accesses_layer = 0

     def access_component(self, component_id):
          # check if it's already there
          self.component_accesses_layer += 1
          if not self.component_status[component_id]:
               self.add_component(component_id)
               self.component_misses_layer += 1


     # note this can only be called from add component b/c it doesn't have its own 
     # internal way of changing the size of stuff stored in the SRAM/updating whether
     # or not it is full 
     def remove_component(self):
          for component_id in range(self.num_component):
               if self.component_status[component_id]:
                    self.component_status[component_id] = 0
                    break

     def add_component(self, component_id):
          if self.name == "input":
               x = 1
          if self.full:
               self.remove_component() # remove component

          else:
               self.fill_size += 1 # increase fill count 
               if self.fill_size == self.memory_size_num_component:
                    self.full = 1

          self.component_status[component_id] = 1 # add component 
          self.DRAM_reads_layer += self.component_size_words # increase DRAM reads

     def conclude_layer(self):
          self.DRAM_reads.append(self.DRAM_reads_layer)
          self.component_misses.append(self.component_misses_layer)
          self.component_accesses.append(self.component_accesses_layer)

     def conclude_NN(self):
          self.DRAM_reads = []