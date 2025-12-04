import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt

def salvar_imagem(vetor_f, largura, altura, nome_arquivo="imagem_reconstruida.png"):
    if len(vetor_f) != largura * altura:
        raise ValueError("O tamanho do vetor 'f' não corresponde às dimensões da imagem.")

    imagem_matrix = vetor_f.reshape((altura, largura))

    plt.imsave(nome_arquivo, imagem_matrix, cmap='gray')
    print(f"\nImagem salva com sucesso como '{nome_arquivo}'")

def carregar_ou_criar_npy(caminho_csv, delimiter=','):
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

def cgnr(H, g, max_iter=10, tol=1e-4):
    print("\n--- Iniciando Algoritmo CGNR em Python (Conforme Moodle) ---")
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

        w_dot_w = np.dot(w, w)

        alpha = z_dot_z_old / w_dot_w

        f = f + alpha * p
        r = r - alpha * w

        r_dot_r_new = np.dot(r, r)
        epsilon = abs(r_dot_r_new - r_dot_r_old)

        if epsilon < tol and i > 0:
            print(f"Convergência atingida na iteração {iterations_done} (erro < {tol})")
            break

        z = H.T @ r
        z_dot_z_new = np.dot(z, z)
        beta = z_dot_z_new / z_dot_z_old
        p = z + beta * p

        z_dot_z_old = z_dot_z_new
        r_dot_r_old = r_dot_r_new

    end_time = time.time()
    print(f"Execução finalizada após {iterations_done} iterações.")

    duration = end_time - start_time

    return {
        "imagem_f": f,
        "iteracoes": iterations_done,
        "tempo_s": duration
    }

if __name__ == '__main__':
    print("--- Iniciando a execução do teste principal ---")
    try:
        caminho_matriz_h = r'../Img1/H-1.csv'
        caminho_sinal_g  = r'../Img1/G-1.csv'

        print(f"Carregando Matriz H de: {caminho_matriz_h}")
        H = carregar_ou_criar_npy(caminho_matriz_h, delimiter=',')

        print(f"Carregando Sinal g de: {caminho_sinal_g}")
        g = carregar_ou_criar_npy(caminho_sinal_g, delimiter=',')

        # --- ADICIONE ESTAS LINHAS DE DEBUG AQUI ---
        print("\n--- TESTE DE SANIDADE DO VETOR G ---")
        print(f"Total de elementos em g: {g.size}")
        print(f"Valor MÁXIMO em g: {np.max(g)}")
        print(f"Valor MÍNIMO em g: {np.min(g)}")
        print(f"Soma de todos os valores em g: {np.sum(g)}")
        if np.sum(g) == 0.0:
            print(">>> ALERTA: A SOMA DE 'g' É ZERO. O ARQUIVO ESTÁ CORROMPIDO OU É O ERRADO. <<<")
        print("------------------------------------\n")
        print("\nAplicando ganho de sinal (γ) ao vetor g...")
        S = 794  # Número de amostras do sinal (para imagem 60x60)
        N = 64   # Número de elementos sensores (para imagem 60x60)

        g_modificado = g.copy()

        # Evitar erro se g for 1D
        if g_modificado.shape == (S * N,):
             g_reshaped = g_modificado.reshape((S, N))
        elif g_modificado.shape == (S, N):
             g_reshaped = g_modificado
        else:
             raise ValueError(f"Vetor g tem shape inesperado: {g_modificado.shape}")

        for l in range(S):
            gamma_l = np.sqrt(100 + (1/20) * l * l)
            g_reshaped[l, :] = g_reshaped[l, :] * gamma_l

        g_modificado = g_reshaped.flatten()

        print("Ganho de sinal aplicado.")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024) # Memória em MB

        resultado = cgnr(H, g_modificado, max_iter=10, tol=1e-4)

        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        print("\n========== RELATÓRIO DE DESEMPENHO (PYTHON) ==========")
        print(f"Iterações executadas....: {resultado['iteracoes']}")
        print(f"Tempo total de execução.: {resultado['tempo_s']:.4f} segundos")
        print(f"Memória RAM utilizada...: {mem_used:.2f} MB")
        print(f"Tamanho da imagem.......: {int(np.sqrt(len(resultado['imagem_f'])))} x {int(np.sqrt(len(resultado['imagem_f'])))} pixels")
        print("======================================================")

        imagem_f = resultado['imagem_f']

        f_min = imagem_f.min()
        f_max = imagem_f.max()

        if (f_max - f_min) < 1e-12:
            imagem_f_normalizada = np.zeros_like(imagem_f)
        else:
            imagem_f_normalizada = (imagem_f - f_min) / (f_max - f_min)

        salvar_imagem(imagem_f_normalizada, 60, 60, nome_arquivo="resultado_imagem_1_py_RUIDOSA.png")

        threshold = 0.465
        imagem_f_limpa = np.where(imagem_f_normalizada < threshold, 0.0, imagem_f_normalizada)

        salvar_imagem(imagem_f_limpa, 60, 60, nome_arquivo="resultado_imagem_1_py_LIMPA.png")


    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: {e}")
        print("Por favor, verifique se o nome e o caminho do arquivo no código estão corretos.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")