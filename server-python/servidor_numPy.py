import numpy as np
import time
import os
import psutil
import matplotlib
import threading
from scipy.ndimage import maximum_filter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

MAX_SIMULTANEOUS_TASKS = 4
semaforo_processamento = threading.Semaphore(MAX_SIMULTANEOUS_TASKS)

MIN_RAM_MB_LIVRE = 500.0

def verificar_memoria_disponivel():
    mem = psutil.virtual_memory()
    mem_livre_mb = mem.available / (1024 * 1024)
    if mem_livre_mb < MIN_RAM_MB_LIVRE:
        print(f"   [ALERTA] Memória Baixa! Livre: {mem_livre_mb:.2f}MB")
        return False
    return True

def salvar_imagem_com_dados(vetor_f, largura, altura, nome_arquivo, info_dict, aplicar_limpeza=False, threshold='auto'):
    try:
        vetor_sanitizado = np.nan_to_num(vetor_f, nan=0.0, posinf=1.0, neginf=0.0)

        imagem_matrix = vetor_sanitizado.reshape((altura, largura))
        f_min = imagem_matrix.min()
        f_max = imagem_matrix.max()
        imagem_normalizada = np.zeros_like(imagem_matrix)
        if (f_max - f_min) > 1e-12:
            imagem_normalizada = (imagem_matrix - f_min) / (f_max - f_min)

        imagem_final = imagem_normalizada

        if aplicar_limpeza:
            threshold_val = 0.0
            if threshold == 'auto':
                threshold_val = np.percentile(imagem_normalizada, 98.0)
            else:
                threshold_val = threshold

            imagem_final = np.where(imagem_normalizada < threshold_val, 0.0, imagem_normalizada)

            local_max = maximum_filter(imagem_final, size=3)
            mask = (imagem_final == local_max)
            imagem_final = imagem_final * mask

        fig = plt.figure(figsize=(6, 7))
        plt.imshow(imagem_final, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Algoritmo: {info_dict['algo']}\nImg: {info_dict['nome_base']} ({largura}x{altura})", fontsize=12, fontweight='bold')

        texto_inferior = (
            f"Inicio: {info_dict['inicio']}\n"
            f"Fim:    {info_dict['fim']}\n"
            f"Iteracoes: {info_dict['iter']} | Tempo: {info_dict['tempo_s']:.4f}s\n"
            f"Erro Final: {info_dict['erro']:.2e}"
        )
        plt.xlabel(texto_inferior, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=100)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"ERRO ao salvar imagem: {e}")
        plt.close('all')
        return False

def carregar_ou_criar_npy(caminho_csv):
    caminho_npy = caminho_csv.replace('.csv', '.npy')
    if os.path.exists(caminho_npy):
        return np.load(caminho_npy)
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")
    dados = np.loadtxt(caminho_csv, delimiter=',')
    np.save(caminho_npy, dados)
    return dados

def cgnr(H, g, max_iter=10, tol=1e-4):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    f = np.zeros(H.shape[1])
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    r_dot_r_old = np.dot(r, r)
    z_dot_z_old = np.dot(z, z)
    iterations_done = 0
    erro_final = 0.0

    for i in range(max_iter):
        iterations_done = i + 1
        w = H @ p
        w_dot_w = np.dot(w, w)
        if w_dot_w < 1e-20: break
        alpha = z_dot_z_old / w_dot_w
        f = f + alpha * p
        r = r - alpha * w
        r_dot_r_new = np.dot(r, r)
        epsilon = abs(r_dot_r_new - r_dot_r_old)
        erro_final = epsilon
        if epsilon < tol and i > 0: break
        z = H.T @ r
        z_dot_z_new = np.dot(z, z)
        beta = z_dot_z_new / z_dot_z_old
        p = z + beta * p
        z_dot_z_old = z_dot_z_new
        r_dot_r_old = r_dot_r_new

    end_time = time.time()
    mem_used_mb = (process.memory_info().rss - mem_before) / (1024 * 1024)
    return { "imagem_f": f, "iteracoes": iterations_done, "tempo_s": end_time - start_time, "memoria_mb": mem_used_mb, "erro_final": erro_final }

@app.route('/reconstruir', methods=['POST'])
def api_reconstruir():
    if not request.json:
        return jsonify({"status": "erro"}), 400

    verificar_memoria_disponivel()

    semaforo_processamento.acquire()

    try:
        print(f"   [SRV] Processando tarefa: {request.json.get('nome_arquivo_base')}...")

        config = request.json
        ts_inicio = datetime.now().strftime('%d/%m %H:%M:%S')

        H = carregar_ou_criar_npy(config['caminho_h'])
        g = carregar_ou_criar_npy(config['caminho_g'])

        S, N = int(config['s']), int(config['n'])
        g_work = g.reshape((S, N)) if g.shape[0] == S*N else g.copy()

        for l in range(S):
            gamma = np.sqrt(100 + (l**2)/20)
            g_work[l, :] *= gamma
        g_flat = g_work.flatten()

        res = cgnr(H, g_flat)

        ts_fim = datetime.now().strftime('%d/%m %H:%M:%S')

        info_img = {
            "algo": "CGNR (Python)",
            "nome_base": config.get('nome_arquivo_base'),
            "iter": res['iteracoes'],
            "tempo_s": res['tempo_s'],
            "inicio": ts_inicio,
            "fim": ts_fim,
            "erro": res['erro_final'],
            "memoria_mb": res['memoria_mb']
        }

        nome_limpa = f"py_out_{config.get('nome_arquivo_base')}_FINAL.png"
        salvar_imagem_com_dados(res['imagem_f'], int(config['largura']), int(config['altura']),
                                nome_limpa, info_img, aplicar_limpeza=True)

        return jsonify({
            "status": "sucesso",
            "imagem_gerada": nome_limpa,
            "tempo_reconstrucao_s": res['tempo_s'],
            "iteracoes": res['iteracoes'],
            "memoria_mb": res['memoria_mb']
        })

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

    finally:
        semaforo_processamento.release()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)