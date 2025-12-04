#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

// Informa ao 'stb_image_write' para incluir o código de implementação
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Usar namespaces para limpeza (como 'import numpy as np')
using namespace Eigen;
using namespace std::chrono;

// Define os tipos de dados que usaremos (como np.float64)
using MatrixRowMajor = Matrix<double, Dynamic, Dynamic, RowMajor>;
using VectorD = Vector<double, Dynamic>;
using MatrixD = Matrix<double, Dynamic, Dynamic>;


/**
 * @brief Carrega uma matriz de um arquivo CSV.
 * O CSV é lido em uma matriz RowMajor, pois os dados do Python/Numpy
 * são armazenados linha por linha.
 */
MatrixRowMajor loadCSV(const std::string& path) {
    std::cout << "Carregando e processando pela primeira vez: " << path << std::endl;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("ERRO: Nao foi possivel abrir o arquivo: " + path);
    }

    std::string line, cell;
    std::vector<double> values;
    int rows = 0;
    int cols = 0;

    while (std::getline(file, line)) {
        rows++;
        std::stringstream ss(line);
        int line_cols = 0;
        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stod(cell));
            line_cols++;
        }
        if (rows == 1) {
            cols = line_cols;
        }
    }
    file.close();

    // Mapeia o vetor 1D de dados em uma Matriz Eigen (RowMajor)
    return Map<MatrixRowMajor>(values.data(), rows, cols);
}


/**
 * @brief Salva um vetor de imagem (normalizado para 0-1) como um PNG.
 */
void savePng(const VectorD& f_image, int width, int height, const std::string& filename) {
    std::vector<unsigned char> img_data(width * height);
    for (int i = 0; i < f_image.size(); ++i) {
        // Converte de [0.0, 1.0] para [0, 255]
        img_data[i] = static_cast<unsigned char>(std::clamp(f_image(i) * 255.0, 0.0, 255.0));
    }

    // Salva o PNG (1 canal = grayscale)
    if (stbi_write_png(filename.c_str(), width, height, 1, img_data.data(), width) == 0) {
        throw std::runtime_error("ERRO: Falha ao salvar a imagem: " + filename);
    }
    std::cout << "\nImagem salva com sucesso como '" << filename << "'" << std::endl;
}

/**
 * @brief Normaliza um vetor de imagem para a faixa [0.0, 1.0].
 */
VectorD normalizeImage(VectorD f) {
    double f_min = f.minCoeff();
    double f_max = f.maxCoeff();
    double range = f_max - f_min;

    if (range < 1e-12) {
        return VectorD::Zero(f.size());
    }
    // .array() permite operações elemento-a-elemento
    return (f.array() - f_min) / range;
}

/**
 * @brief Aplica o threshold, zerando valores abaixo do limiar.
 */
VectorD applyThreshold(const VectorD& f_normalized, double threshold) {
    // Equivalente a: np.where(f_normalized < threshold, 0.0, f_normalized)
    return (f_normalized.array() < threshold)
        .select(VectorD::Constant(f_normalized.size(), 0.0), f_normalized);
}


// Estrutura para retornar os resultados (como o dicionário Python)
struct CgnrResult {
    VectorD f;
    int iterations;
    double duration_s;
};

/**
 * @brief Implementação do algoritmo CGNR em C++ com Eigen.
 */
CgnrResult cgnr(const MatrixD& H, const VectorD& g, int max_iter = 10, double tol = 1e-4) {
    std::cout << "\n--- Iniciando Algoritmo CGNR em C++ ---" << std::endl;
    auto start_time = high_resolution_clock::now();

    // f0 = 0
    VectorD f = VectorD::Zero(H.cols());
    // r0 = g - H*f0
    VectorD r = g - H * f;
    // z0 = H.T * r0
    VectorD z = H.transpose() * r;
    // p0 = z0
    VectorD p = z;

    // r_dot_r_old = r.T * r
    double r_dot_r_old = r.dot(r);
    // z_dot_z_old = z.T * z
    double z_dot_z_old = z.dot(z);

    int iterations_done = 0;
    for (int i = 0; i < max_iter; ++i) {
        iterations_done = i + 1;

        // wi = H*pi
        VectorD w = H * p;

        // w_dot_w = w.T * w
        double w_dot_w = w.dot(w);

        // Checagem de estabilidade (evita divisão por zero)
        if (w_dot_w < 1e-12) {
            std::cout << "Instabilidade (w_dot_w e zero) na iteracao " << iterations_done << ". Parando." << std::endl;
            break;
        }

        // alpha = z_dot_z_old / w_dot_w
        double alpha = z_dot_z_old / w_dot_w;

        // f = f + alpha * p
        f = f + alpha * p;
        // r = r - alpha * w
        r = r - alpha * w;

        // Critério de parada (epsilon)
        double r_dot_r_new = r.dot(r);
        double epsilon = std::abs(r_dot_r_new - r_dot_r_old);
        if (epsilon < tol && i > 0) {
            std::cout << "Convergencia atingida na iteracao " << iterations_done << " (erro < " << tol << ")" << std::endl;
            break;
        }

        // z = H.T * r
        z = H.transpose() * r;
        double z_dot_z_new = z.dot(z);

        // Checagem de estabilidade
        if (z_dot_z_old < 1e-12) {
            std::cout << "Instabilidade (z_dot_z_old e zero) na iteracao " << iterations_done << ". Parando." << std::endl;
            break;
        }

        // beta = z_dot_z_new / z_dot_z_old
        double beta = z_dot_z_new / z_dot_z_old;
        // p = z + beta * p
        p = z + beta * p;

        z_dot_z_old = z_dot_z_new;
        r_dot_r_old = r_dot_r_new;
    }

    auto end_time = high_resolution_clock::now();
    duration<double> duration = end_time - start_time;
    std::cout << "Execucao finalizada apos " << iterations_done << " iteracoes." << std::endl;

    return {f, iterations_done, duration.count()};
}

// ==============================================================================
// ========================== FUNÇÃO PRINCIPAL (MAIN) ===========================
// ==============================================================================

int main() {
    std::cout << "--- Iniciando a execucao do teste principal ---" << std::endl;
    try {
        const std::string h_path = "../Img1/H-1.csv";
        const std::string g_path = "../Img1/G-1.csv";

        std::cout << "Carregando Matriz H de: " << h_path << std::endl;
        MatrixD H = loadCSV(h_path); // Eigen é ColMajor por padrao, mas nosso CSV é RowMajor

        std::cout << "Carregando Sinal g de: " << g_path << std::endl;
        MatrixD g_mat = loadCSV(g_path);
        VectorD g = g_mat; // Converte (N, 1) Matrix para (N) Vector

        std::cout << "\n--- TESTE DE SANIDADE DO VETOR G ---" << std::endl;
        std::cout << "Total de elementos em g: " << g.size() << std::endl;
        std::cout << "Valor MAXIMO em g: " << g.maxCoeff() << std::endl;
        std::cout << "Valor MINIMO em g: " << g.minCoeff() << std::endl;
        std::cout << "Soma de todos os valores em g: " << g.sum() << std::endl;
        std::cout << "------------------------------------\n" << std::endl;

        std::cout << "Dimensoes de H: (" << H.rows() << ", " << H.cols() << ")" << std::endl;
        std::cout << "Dimensoes de g: (" << g.size() << ")" << std::endl;

        std::cout << "\nAplicando ganho de sinal (gamma) ao vetor g..." << std::endl;
        const int S = 794;
        const int N = 64;

        VectorD g_modificado = g;
        // Mapeia o vetor 'g_modificado' para uma matriz (S, N)
        // Isso permite modificar 'g_modificado' como se fosse uma matriz 2D
        Map<Matrix<double, S, N, RowMajor>> g_reshaped(g_modificado.data());

        for (int l = 0; l < S; ++l) {
            double gamma_l = std::sqrt(100.0 + (1.0 / 20.0) * l * l);
            // Multiplica a linha 'l' inteira pelo ganho
            g_reshaped.row(l) *= gamma_l;
        }
        // g_modificado foi alterado no lugar, nao e necessario "flatten"

        std::cout << "Ganho de sinal aplicado." << std::endl;

        // --- Medição de Memória (PSUtil) ---
        // TODO: Implementar medicao de memoria dependente de S.O.
        // (Ex: GetProcessMemoryInfo no Windows, /proc/self/status no Linux)
        double mem_used = 0.0; // Valor fixo por enquanto

        // Executa o algoritmo
        CgnrResult resultado = cgnr(H, g_modificado, 10, 1e-4);

        // --- Relatório Final ---
        std::cout << "\n========== RELATORIO DE DESEMPENHO (C++) ==========" << std::endl;
        std::cout << "Iteracoes executadas....: " << resultado.iterations << std::endl;
        std::cout.precision(4);
        std::cout << std::fixed << "Tempo total de execucao.: " << resultado.duration_s << " segundos" << std::endl;
        std::cout << std::fixed << "Memoria RAM utilizada...: " << mem_used << " MB (NAO IMPLEMENTADO)" << std::endl;
        std::cout << "Tamanho da imagem.......: 60 x 60 pixels" << std::endl; // Fixo no código
        std::cout << "======================================================" << std::endl;

        // --- Salvar Imagens ---
        VectorD imagem_f = resultado.f;
        VectorD f_normalized = normalizeImage(imagem_f);
        savePng(f_normalized, 60, 60, "resultado_imagem_1_cpp_RUIDOSA.png");

        const double threshold = 0.465;
        VectorD f_limpa = applyThreshold(f_normalized, threshold);
        savePng(f_limpa, 60, 60, "resultado_imagem_1_cpp_LIMPA.png");


    } catch (const std::exception& e) {
        std::cerr << "\nOcorreu um erro inesperado: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}