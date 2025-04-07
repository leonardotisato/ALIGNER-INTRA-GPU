#ifndef TASK_H
#define TASK_H
#include<map>
#include<vector>

#define USES_GLOBAL 1

#define MATCH 1
#define MISMATCH -2
#define GAP 1

using namespace std;


namespace poa_gpu_utils{

typedef short Edge; 

struct TaskRefs{
	
	int uses_global = USES_GLOBAL;
	
	vector<int> nseq_offsets;// = vector<int>(BDIM);
	int tot_nseq = 0;
	char* sequences;
	vector<int> seq_offsets;

	int* space_exceeded;
	char* result;//[WL * MAXL * BDIM];
	int* res_size;//[BDIM];
	
	int* nseq_offsets_d;
	char* sequences_d;
	char* result_d;
	int* seq_offsets_d;
	int* res_size_d;

	// char* reads_d;
	
	// int* lpo_edge_offsets_d;
	// unsigned char* lpo_letters_d;
	// Edge* lpo_edges_d;	
	// int* edge_bounds_d;
	// unsigned char* end_nodes_d;

	char* dyn_letters_global_d;	
	Edge* dyn_edges_global_d;
	int* dyn_edge_bounds_global_d;
	// unsigned char* dyn_end_nodes_global_d;

	unsigned char* moves_global_d;	
	short* diagonals_global_sc_d;

	int* d_offsets_global_d;
	int* x_to_ys_d;
	int* y_to_xs_d;
	
	int* dyn_len_global_d;	
};

} //end poa_gpu_utils

#endif
	
