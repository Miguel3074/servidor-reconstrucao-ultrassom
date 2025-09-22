import numpy as np # type: ignore
import time

M = np.array([[1, 2], [3, 4]])
N = np.array([[5, 6], [7, 8]])
a = 2
v = np.array([10, 20]) 
print("--- Testando Operações em Python ---")

mn_result = M @ N
print("Resultado de M * N:\n", mn_result)

am_result = a * M
print("\nResultado de a * M:\n", am_result)

ma_result = M @ v
print("\nResultado de M * a:\n", ma_result)


def cgnr(H, g, max_iter=10):
    """
    Implementa o algoritmo CGNR para reconstrução de imagem.
   
    """
    print("\n--- Iniciando Algoritmo CGNR em Python ---")
    start_time = time.time() # Medir o tempo total de execução

    # Inicialização conforme o pseudocódigo
    f = np.zeros(H.shape[1])
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    
    z_dot_z_old = np.dot(z, z)
    
    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        
        w = H @ p
        alpha = z_dot_z_old / np.dot(w, w)
        
        f = f + alpha * p
        r = r - alpha * w
        z = H.T @ r
        
        z_dot_z_new = np.dot(z, z)
        
        # O critério de parada por erro (epsilon) é complexo.
        # Uma verificação padrão de convergência é na norma do resíduo.
        # O requisito principal é parar em 10 iterações.
        
        beta = z_dot_z_new / z_dot_z_old
        p = z + beta * p
        
        z_dot_z_old = z_dot_z_new

    end_time = time.time()
    duration = end_time - start_time

    print(f"Execução em Python finalizada em {duration:.6f} segundos.")
    
    return {
        "imagem_f": f,
        "iteracoes": iterations_done,
        "tempo_s": duration
    }

# --- Exemplo de Uso ---
if __name__ == '__main__':
    # Dados de exemplo para teste (substitua pelos dados reais do projeto)
    # Matriz H (Modelo): S x N (e.g., 50816 x 3600)
    # Vetor g (Sinal): S (e.g., 50816)
    # Imagem f (saída): N (e.g., 3600)
    
    # Criando dados fictícios com dimensões menores para o teste rodar rápido
    S = 100
    N = 20
    H_teste = np.random.rand(S, N)
    g_teste = np.random.rand(S)

    resultado_py = cgnr(H_teste, g_teste)
    print(f"Iterações executadas: {resultado_py['iteracoes']}")
    # print("Vetor da imagem reconstruída (f):", resultado_py['imagem_f'])