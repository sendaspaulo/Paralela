// open_mp_cpu.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// -----------------------------------------------------------------------------
// Projeto: Implementação paralela de K-Means usando OpenMP
// Descrição: Este programa carrega um arquivo CSV contendo amostras numéricas,
// opcionalmente ignora o cabeçalho, e executa o algoritmo de K-Means em paralelo.
// Ele permite configurar o número de clusters (K), o número máximo de iterações,
// o arquivo de dados e o número de threads via defines ou argumentos de linha de comando.
// -----------------------------------------------------------------------------

// ─────────── CONFIGURAÇÃO ───────────────────────────────────────────────────────
// DATA_FILE:      Caminho padrão para o arquivo CSV de entrada
// DEFAULT_K:      Valor default de K (número de clusters)
// DEFAULT_MAX_IT: Valor default de iterações máximas
// SKIP_HEADER:    Define se a primeira linha (header) deve ser ignorada
// NUM_THREADS:    Número de threads a serem usadas pelo OpenMP
// ────────────────────────────────────────────────────────────────────────────────
#define DATA_FILE       "covtype.csv"
#define DEFAULT_K       10
#define DEFAULT_MAX_IT  150
#define SKIP_HEADER     true
#define NUM_THREADS     32   // Ajuste aqui o número de threads para paralelização

// -----------------------------------------------------------------------------
// Função: load_csv
// Objetivo: Lê um arquivo CSV numérico e retorna um vetor de amostras (vetores de double).
// Parâmetro: filename - nome do arquivo CSV a ser carregado
// Detalhes:
//  1. Abre o arquivo e verifica erro de abertura
//  2. Se SKIP_HEADER for true, pula a primeira linha
//  3. Para cada linha:
//     - Remove caracteres de nova linha e vírgulas finais
//     - Divide a linha em tokens separadas por vírgula
//     - Remove espaços em branco em ambos extremos de cada token
//     - Converte cada token para double (ignora tokens não numéricos)
//     - Remove a última coluna (assumida como rótulo)
//     - Adiciona o vetor de valores à estrutura de dados
// Retorno: vetor de vetores de double, cada um representando uma amostra
vector<vector<double>> load_csv(const string& filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "Erro ao abrir arquivo: " << filename << endl;
        exit(1);
    }
    vector<vector<double>> data;
    string line;
#if SKIP_HEADER
    // Se configurado, pula a primeira linha de cabeçalho
    if (!getline(in, line)) {
        cerr << "Arquivo vazio ou sem header para pular" << endl;
        exit(1);
    }
#endif
    // Processa cada linha restante
    while (getline(in, line)) {
        // Remove retornos de carro e vírgulas finais
        while (!line.empty() && (line.back()=='\r' || line.back()=='\n' || line.back()==','))
            line.pop_back();
        if (line.empty()) continue;

        stringstream ss(line);
        vector<double> row;
        string cell;
        // Para cada célula separada por vírgula
        while (getline(ss, cell, ',')) {
            // Remove espaços em branco no início
            cell.erase(cell.begin(),
                       find_if(cell.begin(), cell.end(),
                               [](unsigned char c){ return !isspace(c); }));
            // Remove espaços em branco no fim
            cell.erase(find_if(cell.rbegin(), cell.rend(),
                               [](unsigned char c){ return !isspace(c); }).base(),
                       cell.end());
            try {
                row.push_back(stod(cell)); // Converte string para double
            } catch (invalid_argument&) {
                // Ignora tokens não numéricos
            }
        }
        // Remove a última coluna (rótulo), se existir
        if (!row.empty()) row.pop_back();
        // Adiciona a amostra ao conjunto de dados
        if (!row.empty()) data.emplace_back(move(row));
    }
    return data;
}

// -----------------------------------------------------------------------------
// Função: euclid
// Objetivo: Calcula a distância Euclidiana entre dois vetores de mesma dimensão
// Parâmetros:
//  a - primeiro vetor
//  b - segundo vetor
// Retorno: distância Euclidiana (sqrt da soma dos quadrados das diferenças)
double euclid(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// -----------------------------------------------------------------------------
// Função principal: main
// Descrição: Configura o ambiente OpenMP, carrega dados, inicializa centróides,
// executa o loop principal do K-Means e imprime os resultados.
int main(int argc, char* argv[]) {
    // Configura o número de threads para o OpenMP
    omp_set_num_threads(NUM_THREADS);
    cout << "Número de threads: " << omp_get_max_threads() << endl;

    // Processa argumentos de linha de comando: K, iteracoes e arquivo
    int K = (argc >= 2 ? stoi(argv[1]) : DEFAULT_K);
    int max_iter = (argc >= 3 ? stoi(argv[2]) : DEFAULT_MAX_IT);
    string filename = (argc >= 4 ? argv[3] : DATA_FILE);

    // Carrega os dados do CSV
    auto data = load_csv(filename);
    int N = static_cast<int>(data.size());
    if (N == 0) {
        cerr << "Nenhuma amostra carregada de " << filename << endl;
        return 1;
    }
    int D = static_cast<int>(data[0].size());
    cout << "→ Carreguei " << N << " amostras de " << filename
         << " (dim=" << D << ")" << endl;

    // Valida valor de K
    if (K <= 0 || K > N) {
        cerr << "Valor de K inválido: " << K << endl;
        return 1;
    }

    // Inicializa centróides escolhendo amostras aleatórias distintas
    vector<vector<double>> centroids(K, vector<double>(D));
    mt19937_64 rng(1234);  // Semente fixa para reprodutibilidade
    uniform_int_distribution<int> pick(0, N-1);
    unordered_set<int> used_indices;
    for (int k = 0; k < K; ) {
        int idx = pick(rng);
        if (used_indices.insert(idx).second) {
            centroids[k++] = data[idx];
        }
    }

    // Vetor de rótulos para cada amostra
    vector<int> labels(N, -1);
    // Loop principal do K-Means
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        // Etapa 1: Atribuição de cada ponto ao centróide mais próximo (em paralelo)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double best_dist = numeric_limits<double>::infinity();
            int best_k = 0;
            for (int k = 0; k < K; k++) {
                double dist = euclid(data[i], centroids[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            if (labels[i] != best_k) {
                labels[i] = best_k;
                changed = true;
            }
        }
        // Se não houve mudança nos rótulos, considera convergido
        if (!changed) {
            cout << "Convergiu em " << iter << " iterações." << endl;
            break;
        }
        // Etapa 2: Recalcula os centróides como média dos pontos atribuídos
        vector<vector<double>> sum(K, vector<double>(D, 0.0));
        vector<int> count(K, 0);
        #pragma omp parallel
        {
            // Estruturas locais por thread para evitar contenção
            vector<vector<double>> local_sum(K, vector<double>(D, 0.0));
            vector<int> local_count(K, 0);
            #pragma omp for nowait
            for (int i = 0; i < N; i++) {
                int c = labels[i];
                local_count[c]++;
                for (int d = 0; d < D; d++) {
                    local_sum[c][d] += data[i][d];
                }
            }
            // Região crítica para agregar resultados locais
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    count[k] += local_count[k];
                    for (int d = 0; d < D; d++) {
                        sum[k][d] += local_sum[k][d];
                    }
                }
            }
        }
        // Atualiza cada centróide dividindo pela quantidade de pontos
        for (int k = 0; k < K; k++) {
            if (count[k] == 0) continue; // evita divisão por zero
            for (int d = 0; d < D; d++) {
                centroids[k][d] = sum[k][d] / count[k];
            }
        }
    }

    // Saída final dos centróides e tamanhos dos clusters
    cout << fixed << setprecision(4);
    for (int k = 0; k < K; k++) {
        cout << "Centróide " << k << ": ";
        for (double v : centroids[k]) {
            cout << v << " ";
        }
        cout << endl;
    }
    vector<int> cluster_size(K, 0);
    for (int label : labels) {
        cluster_size[label]++;
    }
    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k << " tem " << cluster_size[k] << " pontos" << endl;
    }

    return 0;
}