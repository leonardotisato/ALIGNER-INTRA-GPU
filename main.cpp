// Genera grafo "complesso" (ora genera una sequenza) -> leggilo da file
// Gestisci meglio generazione reads e collocazione nelle window -> leggile da file
// Capisci meglio come avvengono gli allineamenti: differenza tra window e batch, etc...
// Modifica TUTTO in modo da allocare solo lo spazio necessario

// Traceback si o no? Linear gap penalty o affine gap penalty?

#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <map>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include "include/poa.h"

#define EDGE_F 2 // Heuristic mean degree for graphs

using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now()


int main(int argc, char* argv[]) {

	bool read_from_file = false;
	char* path_ref;	
	
	if(argc < 3){
		cout << "Invalid arguments. Call this program as: ./poa maxSeqSize maxWindowSize sampleSize [read_file.pow]" << endl;
		return 0;
	}

	if(argc == 4){
		read_from_file = true;
		path_ref = argv[3];
	}
	
	string max_seq_size = argv[1];
	const int MAX_SEQ_LEN = check_input_int(max_seq_size);		
	
	string max_w_size = argv[2];
	const int NUM_BLOCKS = check_input_int(max_w_size);		

	if(MAX_SEQ_LEN < 0){
		cout << "Invalid max sequence length provided" << endl;
		return 0;
	}
	if(NUM_BLOCKS < 0){
		cout << "Invalid max window size provided" << endl;
		return 0;
	}

	vector<vector<string>> reads;
	
	if(read_from_file){
		string filepath(path_ref);
		cout << "*** ATTEMPTING TO READ FROM " << filepath << endl;
		read_batch(reads, NUM_BLOCKS, filepath);
		
	}else{
		cerr << "Invalid file path provided";
	}

	print_reads(reads);

	int numReads = 0;
	for(int i = 0; i < reads.size(); i++) {
		numReads += reads[i].size();
	}

	int batchSize = (numReads - 1) / NUM_BLOCKS + 1;

	//SIMPLE GPU EXECUTION SINGLE KERNEL
	int c = 0;

	get_bmean_batch_result_gpu(reads, c, NUM_BLOCKS, batchSize);

	
	return 0;
}