// Genera grafo "complesso" (ora genera una sequenza) -> leggilo da file
// Gestisci meglio generazione reads e collocazione nelle window -> leggile da file
// Capisci meglio come avvengono gli allineamenti: differenza tra window e batch, etc...
// Modifica TUTTO in modo da allocare solo lo spazio necessario

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

#define EDGE_F 3 // Heuristic mean degree for graphs
#define N_THREADS 64

#define MIN_SLEN 1
#define MIN_WLEN 2


using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now()

typedef uint32_t kmer;

// constexpr unsigned int n_threads = 80;

void read_batch_2(vector<vector<string>> &batch, size_t size, string filename){

    std::ifstream infile(filename);
    std::string line;
    int n = 0;
    int i = 0;
    while (getline(infile, line))
    {
        if (n == 0)
        {
            n = stoi(line);
            batch.emplace_back(std::vector<std::string>());
        }
        else
        {
            batch.back().push_back(line);
            n--;
        }
	i++;
    }

}

int check_input_int(string &arg){
	
	try{
		size_t pos;
		int arg_i = stoi(arg, &pos);
		if(pos < arg.size()){
			std::cerr << "Trailing characters after number: " << arg << '\n';
		}
		return arg_i;
	} catch (invalid_argument const &ex) {
		std::cerr << "Invalid number: " << arg << '\n';
		return -1;
	} catch (out_of_range const &ex) {
		std::cerr << "Number out of range: " << arg << '\n';
		return -1;
	}
	
}

int main(int argc, char* argv[]) {

	bool read_from_file = false;
	char* path_ref;	
	
	if(argc < 4){
		cout << "Invalid arguments. Call this program as: ./poa maxSeqSize maxWindowSize sampleSize [read_file.pow]" << endl;
		return 0;
	}

	if(argc == 5){
		read_from_file = true;
		path_ref = argv[4];
	}
	
	string max_seq_size = argv[1];
	const int SEQ_LEN = check_input_int(max_seq_size);		
	
	string max_w_size = argv[2];
	const int WLEN = check_input_int(max_w_size);		
	
	string sample_size = argv[3];
	const int N_ALIGNMENTS = check_input_int(sample_size);		

	if(SEQ_LEN < 0){
		cout << "Invalid max sequence length provided" << endl;
		return 0;
	}
	if(WLEN < 0){
		cout << "Invalid max window size provided" << endl;
		return 0;
	}
	if(N_ALIGNMENTS < 0){
		cout << "Invalid max window size provided" << endl;
		return 0;
	}
	vector<vector<string>> input;
	
	if(read_from_file){
		string filepath(path_ref);
		cout << "*** ATTEMPTING TO READ FROM " << filepath << " SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		read_batch_2(input, N_ALIGNMENTS, filepath);
		cout << "Read " << input.size() << " alignments" << endl;
	}else{
	//	cout << "*** GENERATING RANDOM SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		input = get_random_sample(N_ALIGNMENTS, WLEN, MIN_WLEN, SEQ_LEN, MIN_SLEN);
	}
		
	vector<vector<string>> result_GPU;

	//SIMPLE GPU EXECUTION SINGLE KERNEL
	int c = 0;

	get_bmean_batch_result_gpu(input, result_GPU, c, SEQ_LEN, WLEN);

	
	return 0;
}
