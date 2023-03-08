import pseudo_analytical_sim
import pandas as pd

def runSim():
	NNLayers = setNN()
	hardwareArch = setHardware()

	AMsimulator = pseudo_analytical_sim.hardware_state()
	AMsimulator.set_hardware(hardwareArch)
	AMsimulator.set_NN(NNLayers)
	AMResults = AMsimulator.run_all_layers()

	AMsimulator.print_NN_results()
	#print(AMResults)


def setHardware():
	names = ["Systolic Array Rows", "Systolic Array Cols", "SRAM Input Size", "SRAM Filter Size", "SRAM Output Size", "Batch Size"]
	arrayRows = 5
	arrayCols = 5
	SRAMInputSize = 1000
	SRAMFilterSize = 1000
	SRAMOutputSize = 1000
	batchSize = 1

	hardware = pd.DataFrame([arrayRows, arrayCols, SRAMInputSize, SRAMFilterSize, SRAMOutputSize, batchSize], names)
	return(hardware)

def setNN():
	names = ["Input Rows", "Input Columns", "Filter Rows", "Filter Columns", "Channels", "Num Filter", "X Stride", "Y Stride"]
	inputRows = [5]
	inputCols = [5]
	filterRows = [3]
	filterCols = [3]
	channels = [1]
	numFilter = [6]
	xStride = [1]
	yStride = [1]

	NNLayersAll = []

	for i in range(len(inputRows)):
		NNLayer = pd.DataFrame([inputRows[i], inputCols[i], filterRows[i], filterCols[i], channels[i], numFilter[i], xStride[i], yStride[i]], names)
		NNLayersAll.append(NNLayer)
	return(NNLayersAll)


def main():
	runSim()

			
if __name__ == "__main__":
	print("\n"*5) 
	main()
	print("\n"*5)