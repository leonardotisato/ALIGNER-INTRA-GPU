#include <unordered_map>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <map>
#include "../include/poagpu.cuh"


void get_bmean_batch_result_gpu(vector<vector<string>> reads, int &c){

	poa_gpu_utils::TaskRefs T;		// struct con dentro TUUUUUTTO
	size_t size = reads.size();

	vector<poa_gpu_utils::Task<vector<string>>> gpu_tasks(size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
	
	//task transfer
	int i = 0;
	for(auto s : reads){
		poa_gpu_utils::Task<vector<string>> t = poa_gpu_utils::Task<vector<string>>(i, i, s);
		gpu_tasks[i] = t;
		i++;
	}

	auto start = NOW;

	gpu_POA_alloc(T);
	gpu_POA(gpu_tasks, T);
	gpu_POA_free(T);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();
	std::cout << "Duration: " << c << " microseconds" << std::endl;
}