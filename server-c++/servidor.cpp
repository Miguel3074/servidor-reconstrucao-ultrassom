// g++ servidor.cpp -o servidor.exe -O3 -std=c++17 -lws2_32 -lpsapi -D_WIN32_WINNT=0x0A00
#include <clocale>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <mutex>
#include <condition_variable>

#include "httplib.h"
#include "json.hpp"

#include <windows.h>
#include <psapi.h>

using json = nlohmann::json;
using namespace std;

const size_t LIMITE_MEMORIA_BYTES = 4ULL * 1024 * 1024 * 1024;

class GerenciadorMemoria {
public:
    GerenciadorMemoria(size_t limite_total) : limite_(limite_total), usado_(0) {}

    void adquirir(size_t bytes_necessarios) {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this, bytes_necessarios]() {
            return (usado_ + bytes_necessarios) <= limite_;
        });

        usado_ += bytes_necessarios;

        cout << "   [MEM] Reservado: " << (bytes_necessarios / 1024 / 1024)
             << "MB. Uso Total: " << (usado_ / 1024 / 1024) << "MB" << endl;
    }

    void liberar(size_t bytes_liberados) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (bytes_liberados > usado_) usado_ = 0;
        else usado_ -= bytes_liberados;

        cout << "   [MEM] Liberado: " << (bytes_liberados / 1024 / 1024)
             << "MB. Livre agora." << endl;

        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t limite_;
    size_t usado_;
};

GerenciadorMemoria gerenciador_memoria(LIMITE_MEMORIA_BYTES);

struct MemoryGuard {
    GerenciadorMemoria& manager;
    size_t bytes;

    MemoryGuard(GerenciadorMemoria& m, size_t b) : manager(m), bytes(b) {
        manager.adquirir(bytes);
    }

    ~MemoryGuard() {
        manager.liberar(bytes);
    }
};

string get_current_time_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%d/%m %H:%M:%S", std::localtime(&now_c));
    return string(buf);
}

double get_memory_usage_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
}

double dot_product(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

void vec_add(vector<double>& res, const vector<double>& a, double scale, const vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + scale * b[i];
}

vector<double> mat_vec_mul(const vector<double>& H, int rows, int cols, const vector<double>& v) {
    vector<double> res(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        int row_offset = i * cols;
        for (int j = 0; j < cols; ++j) sum += H[row_offset + j] * v[j];
        res[i] = sum;
    }
    return res;
}

vector<double> mat_t_vec_mul(const vector<double>& H, int rows, int cols, const vector<double>& v) {
    vector<double> res(cols, 0.0);
    for (int i = 0; i < rows; ++i) {
        double vi = v[i];
        int row_offset = i * cols;
        for (int j = 0; j < cols; ++j) res[j] += H[row_offset + j] * vi;
    }
    return res;
}

double norm_sq(const vector<double>& v) {
    return dot_product(v, v);
}

vector<double> carregar_csv(string path) {
    vector<double> data;
    string path_bin = path;
    size_t ext_pos = path_bin.find_last_of(".");
    if (ext_pos != string::npos) path_bin = path_bin.substr(0, ext_pos) + ".bin";
    else path_bin += ".bin";

    ifstream bin_file(path_bin, ios::binary);
    if (bin_file.is_open()) {
        bin_file.seekg(0, ios::end);
        size_t file_size = bin_file.tellg();
        bin_file.seekg(0, ios::beg);
        size_t num_elements = file_size / sizeof(double);
        data.resize(num_elements);
        bin_file.read(reinterpret_cast<char*>(data.data()), file_size);
        return data;
    }

    cout << "   [IO] Lendo CSV... " << path << endl;
    ifstream file(path);
    if (!file.is_open()) return data;

    string line;
    while (getline(file, line)) {
        for (char &c : line) if (c == ';' || c == ',') c = ' ';
        size_t start = 0, end = 0;
        while ((end = line.find_first_of(" \t\r\n", start)) != string::npos) {
            if (end > start) try { data.push_back(stod(line.substr(start, end - start))); } catch (...) {}
            start = end + 1;
        }
        if (start < line.length()) try { data.push_back(stod(line.substr(start))); } catch (...) {}
    }

    if (!data.empty()) {
        ofstream out_bin(path_bin, ios::binary);
        if (out_bin.is_open()) out_bin.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
    }
    return data;
}

bool salvar_pgm(const vector<double>& img, int width, int height, const string& filename, bool limpar) {
    double min_val = 1e9, max_val = -1e9;
    for (double v : img) { if (v < min_val) min_val = v; if (v > max_val) max_val = v; }

    ofstream file(filename);
    if (!file.is_open()) return false;

    vector<double> processada = img;
    double range = max_val - min_val;
    if (range < 1e-12) range = 1.0;

    for (size_t i = 0; i < processada.size(); ++i) processada[i] = (processada[i] - min_val) / range;

    if (limpar) {
        vector<double> sorted_img = processada;
        std::sort(sorted_img.begin(), sorted_img.end());
        int idx_cut = static_cast<int>(sorted_img.size() * 0.97);
        if (idx_cut >= sorted_img.size()) idx_cut = sorted_img.size() - 1;
        double threshold = sorted_img[idx_cut];
        for (size_t i = 0; i < processada.size(); ++i) if (processada[i] < threshold) processada[i] = 0.0;

        vector<double> img_nms(width * height, 0.0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx_atual = y * width + x;
                double val_atual = processada[idx_atual];
                if (val_atual == 0.0) continue;
                double max_vizinho = -1.0;
                for (int ny = y - 1; ny <= y + 1; ++ny) {
                    for (int nx = x - 1; nx <= x + 1; ++nx) {
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            double val_vizinho = processada[ny * width + nx];
                            if (val_vizinho > max_vizinho) max_vizinho = val_vizinho;
                        }
                    }
                }
                if (val_atual >= max_vizinho - 1e-9) img_nms[idx_atual] = val_atual;
            }
        }
        processada = img_nms;
    }

    file << "P2\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file << static_cast<int>(processada[i] * 255.0) << " ";
        if ((i + 1) % width == 0) file << "\n";
    }
    return true;
}

struct ResultadoCGNR {
    vector<double> imagem;
    int iteracoes;
    double tempo_s;
    double erro_final;
    double memoria_mb;
};

ResultadoCGNR executar_cgnr(const vector<double>& H, const vector<double>& g, int rows, int cols, int max_iter = 10, double tol = 1e-4) {
    double mem_start = get_memory_usage_mb();
    auto start = chrono::high_resolution_clock::now();
    vector<double> f(cols, 0.0);
    vector<double> r = g;
    vector<double> z = mat_t_vec_mul(H, rows, cols, r);
    vector<double> p = z;

    double r_norm_sq_old = norm_sq(r);
    double z_norm_sq_old = norm_sq(z);
    int iter = 0;
    double erro_final = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        iter++;
        vector<double> w = mat_vec_mul(H, rows, cols, p);
        double w_norm_sq = norm_sq(w);
        if (w_norm_sq < 1e-20) break;

        double alpha = z_norm_sq_old / w_norm_sq;
        vec_add(f, f, alpha, p);
        vec_add(r, r, -alpha, w);

        double r_norm_sq_new = norm_sq(r);
        double epsilon = std::abs(r_norm_sq_new - r_norm_sq_old);
        erro_final = r_norm_sq_new;

        if (epsilon < tol && i > 0) break;

        z = mat_t_vec_mul(H, rows, cols, r);
        double z_norm_sq_new = norm_sq(z);
        double beta = z_norm_sq_new / z_norm_sq_old;
        vec_add(p, z, beta, p);
        z_norm_sq_old = z_norm_sq_new;
        r_norm_sq_old = r_norm_sq_new;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    double mem_end = get_memory_usage_mb();
    double mem_used = mem_end - mem_start;
    if (mem_used < 0) mem_used = 0;

    return {f, iter, diff.count(), erro_final, mem_used};
}

bool tem_memoria_livre(double min_mb_livre) {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);

    if (GlobalMemoryStatusEx(&memInfo)) {
        double livre_mb = static_cast<double>(memInfo.ullAvailPhys) / (1024.0 * 1024.0);

        if (livre_mb < min_mb_livre) {
            cout << "   [ALERTA] Memoria Baixa! Livre: " << fixed << setprecision(2) << livre_mb << " MB" << endl;
            return false;
        }
        return true;
    }
    return true;
}

int main() {
    httplib::Server svr;
    cout << "--- Servidor C++  ---" << endl;

    svr.Post("/reconstruir", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);

            string nome_base = j["nome_arquivo_base"];
            string path_h = j["caminho_h"];
            string path_g = j["caminho_g"];
            int largura = j["largura"];
            int altura = j["altura"];
            int S = j["s"];
            int N = j["n"];

            size_t num_elementos_H = (size_t)(S * N) * (size_t)(largura * altura);
            size_t bytes_estimados = num_elementos_H * sizeof(double);

            bytes_estimados = (size_t)(bytes_estimados * 1.1);

            cout << "[REQ] " << nome_base << " requer aprox: " << (bytes_estimados / 1024 / 1024) << " MB" << endl;

            MemoryGuard guard(gerenciador_memoria, bytes_estimados);


            string start_time = get_current_time_str();

            cout << "Processando: " << nome_base << "..." << endl;

            vector<double> H = carregar_csv(path_h);
            vector<double> g = carregar_csv(path_g);

            if (H.empty() || g.empty()) throw runtime_error("CSV Vazio");

            size_t tamanho_esperado = (size_t)S * N * largura * altura;
            if (H.size() != tamanho_esperado) throw runtime_error("ERRO TAMANHO H");

            for (int l = 0; l < S; ++l) {
                double gamma = sqrt(100.0 + (l * l) / 20.0);
                for (int c = 0; c < N; ++c) {
                    if ((l * N + c) < g.size()) g[l * N + c] *= gamma;
                }
            }

            ResultadoCGNR resultado = executar_cgnr(H, g, S * N, largura * altura);

            string end_time = get_current_time_str();

            string nome_pgm = "cpp_out_" + nome_base + ".pgm";
            salvar_pgm(resultado.imagem, largura, altura, nome_pgm, true);

            string nome_json_meta = "cpp_out_" + nome_base + ".json";
            json meta;
            meta["algo"] = "CGNR (C++)";
            meta["nome_base"] = nome_base;
            meta["largura"] = largura;
            meta["altura"] = altura;
            meta["inicio"] = start_time;
            meta["fim"] = end_time;
            meta["iteracoes"] = resultado.iteracoes;
            meta["tempo_s"] = resultado.tempo_s;
            meta["erro_final"] = resultado.erro_final;
            meta["memoria_mb"] = resultado.memoria_mb;

            ofstream f_json(nome_json_meta);
            f_json << meta.dump(4);
            f_json.close();

            json resp;
            resp["status"] = "sucesso";
            resp["imagem_gerada_ruidosa"] = nome_pgm;
            resp["iteracoes_executadas"] = resultado.iteracoes;
            resp["tempo_reconstrucao_s"] = resultado.tempo_s;
            resp["memoria_mb"] = resultado.memoria_mb;
            res.set_content(resp.dump(), "application/json");

        } catch (const exception& e) {
            cout << "[ERRO] " << e.what() << endl;
            res.status = 500;
            json erro;
            erro["status"] = "erro";
            erro["mensagem"] = e.what();
            res.set_content(erro.dump(), "application/json");
        }
    });

    svr.listen("127.0.0.1", 5000);
}