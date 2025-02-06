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

#define EDGE_F 3 // Heuristic mean degree for graphs
#define N_THREADS 64

#define MIN_SLEN 1
#define MIN_WLEN 2


using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now()

typedef uint32_t kmer;

// constexpr unsigned int n_threads = 80;

void read_batch_2(vector<vector<string>> &reads, size_t size, string filename){

	ifstream infile(filename);
    int i = 0;

    if (!infile.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        return;
    }

    string line;
    vector<string> readsVector;

    // Lettura del file FASTA o simile
    while (getline(infile, line)) {
        // Rimozione spazi bianchi iniziali e finali
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        if (!line.empty()) {
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
        }

        if (line.empty() || line[0] == '>') continue;

        readsVector.push_back(line);         

		i++;
		if(i % size == 0) {
			reads.push_back(readsVector);
			readsVector.clear();
		}
    }

	if (!readsVector.empty()) {
        reads.push_back(readsVector);
    }

    infile.close();
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
	vector<vector<string>> reads;
	
	if(read_from_file){
		string filepath(path_ref);
		cout << "*** ATTEMPTING TO READ FROM " << filepath << " SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		read_batch_2(reads, /*N_ALIGNMENTS*/ WLEN, filepath);
		cout << "Read " << reads.size() << " alignments" << endl;
	}else{
	//	cout << "*** GENERATING RANDOM SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		reads = get_random_sample(N_ALIGNMENTS, WLEN, MIN_WLEN, SEQ_LEN, MIN_SLEN);
	}

	for(int i = 0; i < reads.size(); i++) {
		for(int j = 0; j < reads[i].size(); j++) {
			cout << reads[i][j] << endl;
		}
	}
		
	vector<vector<string>> result_GPU;

	//SIMPLE GPU EXECUTION SINGLE KERNEL
	int c = 0;

	get_bmean_batch_result_gpu(reads, result_GPU, c /*, SEQ_LEN, WLEN*/);

	
	return 0;
}
