// TODO: 
// - convertitore CSR to CSR di peve (o direttamente GFA to CSR di Peve)
// - ottimizza copia di dati da memoria host a memoria device
// - salva rilìsultato punteggio di allineamento
// - salva cigar
// - gestione MAXL e SL
// - fixare bug dovuto all'incremento di MAXL
// - testa grafo "complesso"
// - forse: rimuovi EDGE_F

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
#include "include/gfaToGraph.h"

#define EDGE_F 2 // Heuristic mean degree for graphs

using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now()


int main(int argc, char* argv[]) {

	bool read_from_file = false;
	char* path_ref;
	char* path_graph;
	
	if(argc < 5){
		cout << "Invalid arguments. Call this program as: ./poa seqSize numBlocks [read_file.pow] [graph_file.gfa]" << endl;
		return 0;
	}

	if(argc == 5){
		read_from_file = true;
		path_ref = argv[3];
		path_graph = argv[4];
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

	graph_h g;

	string filepath(path_graph);
	cout << "*** ATTEMPTING TO READ GRAPH FROM " << filepath << " ***" << endl;
	convertGFAtoGraph(&g, filepath);

	int batchSize = (numReads - 1) / NUM_BLOCKS + 1;

	//SIMPLE GPU EXECUTION SINGLE KERNEL
	int c = 0;

	get_bmean_batch_result_gpu(reads, c, NUM_BLOCKS, batchSize, &g);

	delete[] g.dyn_letters_global;
	delete[] g.dyn_edges_global;
	delete[] g.dyn_edge_bounds_global;
	
	return 0;
}