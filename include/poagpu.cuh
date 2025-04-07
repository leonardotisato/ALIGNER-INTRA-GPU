#ifndef POAGPU_H
#define POAGPU_H

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include <thread>
#include "poa.h"

#define DEBUG 1

#define SL 10
#define MAXL 10

#define EDGE_F 2 // Heuristic mean degree for graphs


using namespace std;
using namespace chrono;
using namespace poa_gpu_utils;

#define NOW high_resolution_clock::now() 

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {

	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


inline void gpu_POA_alloc(TaskRefs &T, const int numBlocks, int batchSize){

	// alloco memoria sul device, la memoria è puntata da puntatori che risiedono sulla memoria dell'host

	T.result = (char*)malloc(batchSize * MAXL * numBlocks);
	T.res_size = (int*)malloc(numBlocks * sizeof(int));

	cudaErrchk(cudaMalloc(&T.space_exceeded, sizeof(int)));

	cudaErrchk(cudaMalloc(&T.dyn_len_global_d, (unsigned long long)numBlocks * sizeof(int)));

	// allocazione dell'array di struct (uno per blocco)
	// cudaErrchk(cudaMalloc((void**)T.g, numBlocks * sizeof(graph_d)));

	// // graph_d* h_graphs = (graph_d*)malloc(numBlocks * sizeof(graph_d));
	// for (int i = 0; i < numBlocks; i++) {
    // //    cudaErrchk(cudaMalloc((void**)&h_graphs[i].lpo_edge_offsets, batchSize * sizeof(int)));
    // //    cudaErrchk(cudaMalloc((void**)&h_graphs[i].lpo_edges, batchSize * EDGE_F * sizeof(Edge)));
    // //    cudaErrchk(cudaMalloc((void**)&h_graphs[i].lpo_letters, batchSize * sizeof(unsigned char)));

	// 	cudaErrchk(cudaMalloc((void**)&T.g[i].lpo_edge_offsets, batchSize * sizeof(int)));
	// 	cudaErrchk(cudaMalloc((void**)&T.g[i].lpo_edges, batchSize * EDGE_F * sizeof(Edge)));
	// 	cudaErrchk(cudaMalloc((void**)&T.g[i].lpo_letters, batchSize * sizeof(unsigned char)));
    // }

	// cudaErrchk(cudaMalloc(&T.reads_d, (unsigned long long)batchSize * SL * numBlocks));

	cudaErrchk(cudaMalloc(&T.sequences_d, (unsigned long long)SL * batchSize * numBlocks)); 
	cudaErrchk(cudaMalloc(&T.seq_offsets_d, (unsigned long long)numBlocks * batchSize * sizeof(int))); 
	cudaErrchk(cudaMalloc(&T.nseq_offsets_d, (unsigned long long)numBlocks * sizeof(int))); 
	cudaErrchk(cudaMalloc(&T.result_d, (unsigned long long)MAXL * batchSize * numBlocks)); 
	cudaErrchk(cudaMalloc(&T.res_size_d, (unsigned long long)numBlocks * sizeof(int)));

	// cudaErrchk(cudaMalloc(&T.lpo_edge_offsets_d, (unsigned long long)batchSize * numBlocks * sizeof(int)));
	// cudaErrchk(cudaMalloc(&T.lpo_letters_d, (unsigned long long)batchSize * SL * numBlocks));
	// cudaErrchk(cudaMalloc(&T.lpo_edges_d, (unsigned long long)batchSize * SL * EDGE_F * numBlocks * sizeof(Edge)));
	// cudaErrchk(cudaMalloc(&T.edge_bounds_d, (unsigned long long)batchSize * (SL+1) * numBlocks * sizeof(int)));	
	// cudaErrchk(cudaMalloc(&T.end_nodes_d, (unsigned long long)batchSize * SL * numBlocks));

	cudaErrchk(cudaMalloc(&T.dyn_letters_global_d, (unsigned long long)MAXL * numBlocks));
	cudaErrchk(cudaMalloc(&T.dyn_edges_global_d, (unsigned long long)MAXL * EDGE_F * numBlocks * sizeof(Edge)));
	cudaErrchk(cudaMalloc(&T.dyn_edge_bounds_global_d, (unsigned long long)(MAXL+1) * numBlocks * sizeof(int)));
	// cudaErrchk(cudaMalloc(&T.dyn_end_nodes_global_d, (unsigned long long)MAXL * numBlocks));
	
	cudaErrchk(cudaMalloc(&T.moves_global_d, (unsigned long long)2 * (MAXL+1) * (SL+1) * numBlocks * sizeof(unsigned char)));
	cudaErrchk(cudaMalloc(&T.diagonals_global_sc_d, (unsigned long long)(MAXL+1)*(SL+1) * numBlocks * sizeof(short)));

	cudaErrchk(cudaMalloc(&T.d_offsets_global_d, (unsigned long long)(MAXL + SL+1) * numBlocks * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.x_to_ys_d, (unsigned long long)MAXL * numBlocks * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.y_to_xs_d, (unsigned long long)MAXL * numBlocks * sizeof(int)));
	//cout << "Alloc completed\n";

	// for (int i = 0; i < numBlocks; i++) {
	// 	cudaFree(h_graphs[i].lpo_edge_offsets);
	// 	cudaFree(h_graphs[i].lpo_edges);
	// 	cudaFree(h_graphs[i].lpo_letters);
	// }
	// free(h_graphs);
}

inline void gpu_POA_free(TaskRefs &T){

	cudaDeviceReset();
	
	free(T.result);
	free(T.res_size);

	// for (int i = 0; i < numBlocks; i++) {
	// 	cudaFree(T.g[i].lpo_edge_offsets);
	// 	cudaFree(T.g[i].lpo_edges);
	// 	cudaFree(T.g[i].lpo_letters);
	// }
	// free(T.g);

}

static const int NOT_ALIGNED = -1;

void init_kernel_block_parameters(vector<vector<string>> &reads, char** sequences, vector<int> &nseq_offsets, 
								vector<int> &seq_offsets, int* tot_nseq, int first_el, const int numBlocks, int batchSize) {

	cout << "FIRST EL = " << first_el << endl;
						
	int n = (first_el + numBlocks < reads.size()) ? numBlocks : reads.size() - first_el;

	// first_el == batch_Idx
	// for dall'inizio del batch alla fine del batch 

	// filling nseq_offset
	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
		
		nseq_offsets[window_idx - first_el] = reads[window_idx].size();
	}
	partial_sum(nseq_offsets.begin(),nseq_offsets.end(),nseq_offsets.begin());


	// printing things
	cout << "N_seq_offs: ";
	for(int i = first_el; i < numBlocks + first_el && i < reads.size(); i++) {
		cout << nseq_offsets[i - first_el] << " ";
	}
	cout << endl;


	*tot_nseq = nseq_offsets[n-1];
	seq_offsets = vector<int>(*tot_nseq);

	cout << "tot seq = " << *tot_nseq << endl;
	
	int sequence_idx = 0;
	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
		vector<string> &window = reads[window_idx];			
		int wsize = window.size();
		for(int i = 0; i < wsize; i++) {
			seq_offsets[sequence_idx] = window[i].size();
			sequence_idx++;
		}
	}
	partial_sum(seq_offsets.begin(), seq_offsets.end(), seq_offsets.begin());

	// printing things (sembrerebbe che seq_offsets sia in termini di numero di caratteri)
	int numReads = 0;
	for(int i = 0; i < reads.size(); i++){
		numReads += reads[i].size();
	}
	cout << "numReads = " << numReads << endl;
	cout << "seq_offs: ";
	// upper bound of the forcycle isn'st correct (reads index should be updated (not as I))
	for(int i = first_el; i < numBlocks * batchSize && i < numReads + first_el && i < first_el + numBlocks * reads[0].size(); i++) {
		cout << seq_offsets[i - first_el] << " ";
	}
	cout << endl;

	// tot size è in numero di caratteri
	int tot_size = seq_offsets[sequence_idx-1];

	cout << "tot_size = " << tot_size << endl;

	*sequences = (char*)malloc(tot_size);
	sequence_idx = 0;

	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
			
		vector<string>& window = reads[window_idx];	
		for(auto seq : window) {
			int offset;
			if(sequence_idx == 0){
				offset = 0;
			}else{
				offset = seq_offsets[sequence_idx-1];
			}
			char* seq_ptr = (*sequences) + offset;
			memcpy(seq_ptr, seq.c_str(), seq.size());
			sequence_idx++;
		}
	}	
}


void gpu_POA(vector<vector<string>> &reads, TaskRefs &T, const int numBlocks, int batchSize) {

	int numReads = 0;
	for(int i = 0; i < reads.size(); i++) {
		numReads += reads[i].size();
	}

	int lastBatch = numReads % numBlocks;

	int *space_exceeded = (int*)malloc(sizeof(int));

	T.nseq_offsets = vector<int>(numBlocks);

	// assegna ai puntatori allocati sul device i puntatori di T
	assign_device_memory<<<1, 1>>>(/*T.lpo_edge_offsets_d, T.lpo_letters_d, T.lpo_edges_d, 
				       T.edge_bounds_d, T.end_nodes_d,*/ T.dyn_letters_global_d, T.dyn_edges_global_d, T.dyn_edge_bounds_global_d, 
				       /*T.dyn_end_nodes_global_d,*/ T.moves_global_d, T.diagonals_global_sc_d, 
                       T.d_offsets_global_d, T.x_to_ys_d, T.y_to_xs_d, T.dyn_len_global_d, numBlocks, T.seq_offsets_d, T.sequences_d);

	cudaStreamSynchronize(0);

	int block_offset = 0;
	int BLOCKS = numBlocks;
	
	init_kernel_block_parameters(reads, &T.sequences, T.nseq_offsets, T.seq_offsets, &T.tot_nseq, block_offset, numBlocks, batchSize); //block_offset=0
	
	cudaErrchk(cudaMemcpy(T.space_exceeded, space_exceeded, sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.sequences_d, T.sequences, T.seq_offsets[T.tot_nseq-1], cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.seq_offsets_d, T.seq_offsets.data(), T.tot_nseq * sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.nseq_offsets_d, T.nseq_offsets.data(), (unsigned long long)BLOCKS * sizeof(int), cudaMemcpyHostToDevice));
	
	//cout << "Compute edge offsets\n";

	// numThread == batchSize ??
	// devo avere numReads / numBlocks threads
	// compute_edge_offsets<<<BLOCKS, batchSize>>>(T.seq_offsets_d, T.nseq_offsets_d);

	// print_lpo_offsets<<<1,1>>>(numBlocks, batchSize);
	
	cudaStreamSynchronize(0);
	
	//cout << "Generate LPO\n";

		// questo sarà da rimuovere
	// for(int i = 0; i < batchSize; i++){
	// 	generate_lpo<<<BLOCKS, SL+1>>>(T.sequences_d, T.seq_offsets_d, T.nseq_offsets_d, i);
	// }
	generate_lpo<<<BLOCKS, SL+1>>>(T.sequences_d, T.seq_offsets_d, T.nseq_offsets_d, 0);

	printGraphStructure<<<1, 1>>>(numBlocks, batchSize);

	cudaStreamSynchronize(0);

	int i_seq_idx = 0;

	for(int j_seq_idx = 1; j_seq_idx < batchSize; j_seq_idx++) {

		if(j_seq_idx == batchSize-1 && lastBatch != 0){
			BLOCKS = lastBatch;
		}

		cout << "BLOCKS = " << BLOCKS << "   numBlocks = " << numBlocks << "   batchSize = " << batchSize <<
				"   j_seq_idx = " << j_seq_idx << "   i_seq_idx = " << i_seq_idx << "   blocks = " << BLOCKS << endl;		
		
		// printf("\n\nGRAPH CREATED --> BEGIN ALIGNMENT\n");

		cudaStreamSynchronize(0);

		// prossimi 2 kernel servono solo e soltanto per la dpMatrix ??
				
		compute_d_offsets<<<BLOCKS, 1>>>(i_seq_idx, j_seq_idx, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0); 
		
		init_diagonals<<<BLOCKS, 1>>>(i_seq_idx, j_seq_idx, T.uses_global, T.nseq_offsets_d);
		
		//cout << "Alignment kernel call\n";
		
		sw_align<<<BLOCKS, SL+1>>>(i_seq_idx, j_seq_idx, T.uses_global, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0);
	}
}


#endif