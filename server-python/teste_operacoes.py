import numpy as np
import time
import os
import psutil 

# ==============================================================================
# FUNÇÃO 1: CARREGAR DADOS DE FORMA EFICIENTE
# ==============================================================================
def carregar_ou_criar_npy(caminho_csv, delimiter=';'):
    """
    Carrega um array NumPy de um arquivo .npy se ele existir.
    Caso contrário, carrega do .csv, cria o .npy para futuras execuções,
    e retorna o array.
    """
    caminho_npy = caminho_csv.replace('.csv', '.npy')
    if os.path.exists(caminho_npy):
        print(f"Carregando dados pré-processados de: {caminho_npy}")
        dados = np.load(caminho_npy)
    else:
        print(f"Carregando e processando pela primeira vez: {caminho_csv}")
        if not os.path.exists(caminho_csv):
            raise FileNotFoundError(f"ERRO: O arquivo CSV original não foi encontrado em: {caminho_csv}")
        dados = np.loadtxt(caminho_csv, delimiter=delimiter)
        print(f"Salvando dados em formato .npy para acesso rápido: {caminho_npy}")
        np.save(caminho_npy, dados)
    return dados

# ==============================================================================
# FUNÇÃO 2: ALGORITMO CGNR
# ==============================================================================
def cgnr(H, g, max_iter=10, tol=1e-4):
    """
    Implementa o algoritmo CGNR para reconstrução de imagem, com critério 
    de parada por erro (tolerância) ou número máximo de iterações.
    """
    print("\n--- Iniciando Algoritmo CGNR em Python ---")
    start_time = time.time()

    f = np.zeros(H.shape[1])
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    
    r_dot_r_old = np.dot(r, r) 
    z_dot_z_old = np.dot(z, z)
    
    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        
        w = H @ p
        alpha = z_dot_z_old / np.dot(w, w)
        
        f = f + alpha * p
        r = r - alpha * w
        
        # Critério de Parada por Erro (epsilon)
        r_dot_r_new = np.dot(r, r)
        epsilon = abs(r_dot_r_new - r_dot_r_old)
        if epsilon < tol:
            print(f"Convergência atingida na iteração {iterations_done} (erro < {tol})")
            break

        z_dot_z_new = np.dot(z, z)
        beta = z_dot_z_new / z_dot_z_old
        p = z + beta * p
        
        z_dot_z_old = z_dot_z_new
        r_dot_r_old = r_dot_r_new

    end_time = time.time()
    duration = end_time - start_time
    print(f"Execução finalizada.")
    
    return {
        "imagem_f": f,
        "iteracoes": iterations_done,
        "tempo_s": duration
    }

# ==============================================================================
# BLOCO PRINCIPAL: ONDE O SCRIPT REALMENTE EXECUTA
# ==============================================================================
if __name__ == '__main__':
    print("--- Iniciando a execução do teste principal ---")
    
    # --------------------------------------------------------------------------
    # >> PASSO IMPORTANTE <<
    # Verifique se os nomes dos arquivos abaixo correspondem EXATAMENTE
    # aos nomes dos arquivos na sua pasta "Dados".
    # --------------------------------------------------------------------------
    try:
        caminho_matriz_h = 'Dados/H_60x60.csv'
        caminho_sinal_g = 'Dados/g_imagem1_60x60.csv'

        H = carregar_ou_criar_npy(caminho_matriz_h, delimiter=';')
        g = carregar_ou_criar_npy(caminho_sinal_g, delimiter=';')

        # Medindo o uso de memória ANTES da execução
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024) # Memória em MB

        # Executando o algoritmo principal
        resultado = cgnr(H, g)

        # Medindo o uso de memória DEPOIS da execução
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        # Apresentando os resultados finais
        print("\n========== RELATÓRIO DE DESEMPENHO (PYTHON) ==========")
        print(f"Iterações executadas....: {resultado['iteracoes']}")
        print(f"Tempo total de execução.: {resultado['tempo_s']:.4f} segundos")
        print(f"Memória RAM utilizada...: {mem_used:.2f} MB")
        print(f"Tamanho da imagem.......: {int(np.sqrt(len(resultado['imagem_f'])))} x {int(np.sqrt(len(resultado['imagem_f'])))} pixels")
        print("======================================================")

    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: {e}")
        print("Por favor, verifique se o nome e o caminho do arquivo no código estão corretos.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")