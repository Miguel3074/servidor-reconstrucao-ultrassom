import numpy as np
import time
import os
import psutil 
import matplotlib.pyplot as plt

def salvar_imagem(vetor_f, largura, altura, nome_arquivo="imagem_reconstruida.png"):
    """
    Converte o vetor da imagem 'f' em uma matriz 2D e a salva como um arquivo de imagem.
    """
    if len(vetor_f) != largura * altura:
        raise ValueError("O tamanho do vetor 'f' não corresponde às dimensões da imagem.")

    # Converte o vetor 1D em uma matriz 2D (imagem)
    imagem_matrix = vetor_f.reshape((altura, largura))

    # Usa matplotlib para salvar a imagem
    plt.imsave(nome_arquivo, imagem_matrix, cmap='gray')
    print(f"\nImagem salva com sucesso como '{nome_arquivo}'")

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

        z = H.T @ r
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

def cgnr_regularizado(H, g, lambda_reg, max_iter=10, tol=1e-8):
    """
    Implementa o algoritmo CGLS (Conjugate Gradient for Least Squares),
    que é a versão correta do CGNR com regularização de Tikhonov.
    """
    print("\n--- Iniciando Algoritmo CGNR Regularizado (CGLS) em Python ---")
    start_time = time.time()

    f = np.zeros(H.shape[1])

    r = g - (H @ f)
    s = (H.T @ r) - (lambda_reg * f)

    p = s.copy()
    gamma = np.dot(s, s)

    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1

        q = H @ p
        delta = np.dot(q, q) + ((lambda_reg**2) * np.dot(p, p))

        if delta == 0:
            break

        alpha = gamma / delta

        f = f + alpha * p   # Atualiza a imagem
        r = r - alpha * q   # Atualiza o resíduo r

        # Recalcula o resíduo s para a próxima iteração
        s_new = (H.T @ r) - (lambda_reg * f)

        gamma_new = np.dot(s_new, s_new)

        # Critério de parada pode ser baseado na mudança de 'f'
        f_change = np.linalg.norm(alpha * p) / np.linalg.norm(f)
        if f_change < tol:
            print(f"Convergência atingida na iteração {iterations_done} (mudança em 'f' < {tol})")
            break

        beta = gamma_new / gamma
        p = s_new + beta * p # Atualiza a direção de busca

        s = s_new
        gamma = gamma_new

    end_time = time.time()
    duration = end_time - start_time
    print(f"Execução finalizada.")

    return {
        "imagem_f": f,
        "iteracoes": iterations_done,
        "tempo_s": duration
    }

if __name__ == '__main__':
    print("--- Iniciando a execução do teste principal ---")
    
    # --------------------------------------------------------------------------
    # >> PASSO IMPORTANTE <<
    # Verifique se os nomes dos arquivos abaixo correspondem EXATAMENTE
    # aos nomes dos arquivos na sua pasta "Dados".
    # --------------------------------------------------------------------------
    try:
        caminho_matriz_h = r'Img1/H-1.csv'
        caminho_sinal_g  = r'Img1/G-1.csv'

        print(f"Carregando Matriz H de: {caminho_matriz_h}")
        H = carregar_ou_criar_npy(caminho_matriz_h, delimiter=',')

        print(f"Carregando Sinal g de: {caminho_sinal_g}")
        g = carregar_ou_criar_npy(caminho_sinal_g, delimiter=',')

        # (Opcional, mas recomendado para depuração)
        # Verifique as dimensões dos arrays carregados
        print(f"Dimensões de H: {H.shape}") # Deve ser algo como (50816, 3600)
        print(f"Dimensões de g: {g.shape}") # Deve ser algo como (50816,)

        print("\nAplicando ganho de sinal (γ) ao vetor g...")
        S = 794  # Número de amostras do sinal (para imagem 60x60)
        N = 64   # Número de elementos sensores (para imagem 60x60)

        g_modificado = g.copy()
        g_reshaped = g_modificado.reshape((S, N))

        for l in range(S):
            gamma_l = np.sqrt(100 + (1/20) * l * l)
            g_reshaped[l, :] = g_reshaped[l, :] * gamma_l

        g_modificado = g_reshaped.flatten()

        print("Ganho de sinal aplicado.")

        print("Calculando o coeficiente de regularização (λ)...")
        lambda_reg = np.max(np.abs(H.T @ g_modificado)) * 0.10
        print(f"Coeficiente λ calculado: {lambda_reg}")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024) # Memória em MB

        # Executando o algoritmo principal
        resultado = cgnr_regularizado(H, g_modificado, lambda_reg, max_iter=9999, tol=1e-4)

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

        imagem_f = resultado['imagem_f']
        f_min = imagem_f.min()
        f_max = imagem_f.max()
        imagem_f_normalizada = (imagem_f - f_min) / (f_max - f_min)
        salvar_imagem(imagem_f_normalizada, 60, 60, nome_arquivo="resultado_imagem_1_py.png")

    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: {e}")
        print("Por favor, verifique se o nome e o caminho do arquivo no código estão corretos.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
