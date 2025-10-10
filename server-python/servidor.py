import numpy as np
import time
import os
from datetime import datetime
from flask import Flask, request, jsonify

# ==============================================================================
# FUNÇÕES DE APOIO E ALGORITMO PRINCIPAL
# (Copiadas do seu script 'teste_operacoes.py')
# ==============================================================================

def carregar_ou_criar_npy(caminho_csv, delimiter=','):
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

def cgnr_regularizado(H, g, lambda_reg, max_iter=10, tol=1e-4):
    """
    Implementa o algoritmo CGLS (Conjugate Gradient for Least Squares),
    que é a versão correta do CGNR com regularização de Tikhonov.
    """
    start_time_dt = datetime.now()
    print("\n--- Iniciando Algoritmo CGNR Regularizado (CGLS) em Python ---")
    start_time_perf = time.perf_counter()

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
        if delta == 0: break
        alpha = gamma / delta
        f = f + alpha * p
        r = r - alpha * q
        s_new = (H.T @ r) - (lambda_reg * f)
        gamma_new = np.dot(s_new, s_new)

        if np.linalg.norm(f) > 0:
            f_change = np.linalg.norm(alpha * p) / np.linalg.norm(f)
            if f_change < tol:
                print(f"Convergência atingida na iteração {iterations_done} (mudança em 'f' < {tol})")
                break
        
        beta = gamma_new / gamma
        p = s_new + beta * p
        s = s_new
        gamma = gamma_new

    end_time_perf = time.perf_counter()
    duration = end_time_perf - start_time_perf
    end_time_dt = datetime.now()
    print(f"Execução finalizada.")

    # Retorna o resultado e os metadados conforme os requisitos
    return {
        "imagem_f": f.tolist(), # Converte para lista para ser enviado como JSON
        "metadata": {
            "algoritmo": "CGNR Regularizado (CGLS) - Python",
            "inicio_reconstrucao": start_time_dt.isoformat(),
            "termino_reconstrucao": end_time_dt.isoformat(),
            "tempo_s": duration,
            "iteracoes": iterations_done,
            "pixels_largura": int(np.sqrt(len(f))),
            "pixels_altura": int(np.sqrt(len(f)))
        }
    }

# ==============================================================================
# LÓGICA DO SERVIDOR FLASK
# ==============================================================================
app = Flask(__name__)

# --- Carregamento Inicial do Modelo (Matriz H) ---
# A matriz H é grande e não muda, então a carregamos uma única vez
# quando o servidor é iniciado para otimizar o desempenho.
print("Iniciando servidor... Carregando Matriz H em memória...")
caminho_matriz_h = r'Img1/H-1.csv'
H_global = carregar_ou_criar_npy(caminho_matriz_h, delimiter=',')
print("Matriz H carregada. Servidor pronto para receber requisições na porta 5000.")


@app.route('/reconstruir', methods=['POST'])
def reconstruir_imagem():
    """
    Endpoint da API que recebe um sinal 'g' e retorna a imagem reconstruída.
    """
    # 1. Valida e extrai o sinal 'g' da requisição JSON
    dados = request.get_json()
    if not dados or 'sinal_g' not in dados:
        return jsonify({"erro": "Sinal 'sinal_g' não encontrado no corpo da requisição"}), 400

    g = np.array(dados['sinal_g'])
    print(f"\n--- Nova Requisição Recebida (Tamanho do sinal g: {g.shape}) ---")
    
    # 2. Executa a lógica de pré-processamento (ganho de sinal e lambda)
    print("Aplicando ganho de sinal (γ)...")
    S, N = 794, 64  # Parâmetros para o modelo de 60x60
    g_reshaped = g.copy().reshape((S, N))
    for l in range(S):
        gamma_l = np.sqrt(100 + (1/20) * l * l)
        g_reshaped[l, :] *= gamma_l
    g_modificado = g_reshaped.flatten()

    print("Calculando o coeficiente de regularização (λ)...")
    lambda_reg = np.max(np.abs(H_global.T @ g_modificado)) * 0.10

    # 3. Chama o algoritmo de reconstrução
    resultado = cgnr_regularizado(H_global, g_modificado, lambda_reg)

    # 4. Retorna o resultado completo (imagem + metadados) para o cliente
    print("Reconstrução concluída. Enviando resposta ao cliente.")
    return jsonify(resultado)

if __name__ == '__main__':
    # Roda o servidor web na porta 5000
    app.run(debug=False, port=5000)