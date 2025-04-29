import sys
import os
import subprocess

def print_help():
    print("""
Uso corretto:
    python run_poagpu.py <param1> <param2> <num_vertici> <num_reads> <len_reads> [--example]

Dove:
    <param1> e <param2>     = Parametri numerici da passare all'eseguibile.
    <num_vertici>           = Numero di vertici (presente nel nome del file grafo).
    <num_reads>             = Numero di reads (valore numerico, verrà convertito in 100, 100K, 100M automaticamente).
    <len_reads>             = Lunghezza delle reads (presente nel nome del file delle reads).
    --example               = (Opzionale) Usa i file di esempio da test/examples/.

Esempio:
    python run_poagpu.py 3 2 1000 5000 150
    python run_poagpu.py 3 2 1000 5000 150 --example
""")

def format_reads_folder(num_reads):
    if num_reads >= 1_000_000:
        return f"{num_reads // 1_000_000}M"
    elif num_reads >= 1_000:
        return f"{num_reads // 1_000}K"
    else:
        return str(num_reads)

def main():
    if len(sys.argv) not in [6, 7]:
        print("[ERRORE] Numero errato di parametri.")
        print_help()
        sys.exit(1)

    use_example = False
    if len(sys.argv) == 7:
        if sys.argv[6] == "--example":
            use_example = True
        else:
            print("[ERRORE] Opzione non riconosciuta.")
            print_help()
            sys.exit(1)

    try:
        param1 = int(sys.argv[1])
        param2 = int(sys.argv[2])
        num_vertici = int(sys.argv[3])
        num_reads = int(sys.argv[4])
        len_reads = int(sys.argv[5])
    except ValueError:
        print("[ERRORE] Tutti i parametri devono essere numeri interi.")
        print_help()
        sys.exit(1)

    # Percorso relativo allo script (che sta in scripts/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Costruzione dei percorsi
    poagpu_exec = os.path.join(base_dir, "poagpu")

    if use_example:
        graph_file = os.path.join(base_dir, "test/examples/graph.gfa")
        reads_file = os.path.join(base_dir, "test/examples/reads.fa")
    else:
        graph_file = os.path.join(base_dir, f"test/graph/graph_{num_vertici}.gfa")
        reads_folder = format_reads_folder(num_reads)
        reads_file = os.path.join(base_dir, f"test/reads/{reads_folder}/reads_{len_reads}.fa")

    # Controllo esistenza file
    if not os.path.isfile(poagpu_exec):
        print(f"[ERRORE] Eseguibile non trovato: {poagpu_exec}")
        sys.exit(1)
    if not os.path.isfile(graph_file):
        print(f"[ERRORE] File grafo non trovato: {graph_file}")
        sys.exit(1)
    if not os.path.isfile(reads_file):
        print(f"[ERRORE] File reads non trovato: {reads_file}")
        sys.exit(1)

    # Comando da eseguire
    cmd = [poagpu_exec, str(param1), str(param2), reads_file, graph_file]

    print(f"Eseguo comando: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] Errore durante l'esecuzione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()