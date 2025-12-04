import math
import time
import os
import csv
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)


def ler_csv_como_matriz(caminho):
    """Lê CSV e retorna lista de listas (Matriz) ou lista (Vetor)."""
    dados = []
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    with open(caminho, 'r') as f:
        for linha in f:
            partes = linha.strip().replace(';', ',').split(',')
            nums = [float(x) for x in partes if x.strip()]
            if nums:
                dados.append(nums)

    return dados

def achatar_lista(lista_de_listas):
    """Transforma [[1,2],[3,4]] em [1,2,3,4]"""
    flat = []
    for sub in lista_de_listas:
        flat.extend(sub)
    return flat


def dot_product(v1, v2):
    """Produto escalar: soma(v1[i] * v2[i])"""
    return sum(a * b for a, b in zip(v1, v2))

def vec_sub(v1, v2):
    """v1 - v2"""
    return [a - b for a, b in zip(v1, v2)]

def vec_add_scaled(v1, v2, scale):
    """v1 + (scale * v2)"""
    return [a + (scale * b) for a, b in zip(v1, v2)]

def mat_vec_mul(H, v):
    """Matriz H (lista de listas) vezes Vetor v"""
    resultado = []
    for linha in H:
        resultado.append(dot_product(linha, v))
    return resultado

def mat_T_vec_mul(H, v):
    """Matriz H Transposta vezes Vetor v"""

    num_linhas = len(H)
    num_cols = len(H[0])

    res = [0.0] * num_cols

    for i in range(num_linhas):
        val_v = v[i]
        linha_h = H[i]
        for j in range(num_cols):
            res[j] += linha_h[j] * val_v

    return res

def norm_sq(v):
    """Norma ao quadrado: ||v||^2"""
    return dot_product(v, v)


def salvar_pgm(imagem_vetor, largura, altura, nome_arquivo):
    """Salva em formato PGM (texto simples)"""
    try:
        v_min = min(imagem_vetor)
        v_max = max(imagem_vetor)
        delta = v_max - v_min
        if delta < 1e-12: delta = 1.0

        pixels = []
        for val in imagem_vetor:
            norm = (val - v_min) / delta
            pixel = int(norm * 255)
            pixels.append(pixel)

        with open(nome_arquivo, 'w') as f:
            f.write("P2\n")
            f.write(f"{largura} {altura}\n")
            f.write("255\n")

            count = 0
            for p in pixels:
                f.write(f"{p} ")
                count += 1
                if count >= largura:
                    f.write("\n")
                    count = 0

        print(f"Imagem PGM salva: {nome_arquivo}")
        return True
    except Exception as e:
        print(f"Erro ao salvar PGM: {e}")
        return False


def cgnr_pure(H, g, max_iter=10, tol=1e-4):
    print(f"\n--- Iniciando CGNR (Pure Python) ---")
    start_time = time.time()

    num_cols = len(H[0])

    f = [0.0] * num_cols

    r = list(g)

    z = mat_T_vec_mul(H, r)
    p = list(z)

    r_norm_sq_old = dot_product(r, r)
    z_norm_sq_old = dot_product(z, z)

    iteracoes = 0
    erro_final = 0.0

    for k in range(max_iter):
        iteracoes = k + 1

        w = mat_vec_mul(H, p)

        w_norm_sq = dot_product(w, w)
        if w_norm_sq < 1e-20: break

        alpha = z_norm_sq_old / w_norm_sq

        # f = f + alpha * p
        f = vec_add_scaled(f, p, alpha)

        # r = r - alpha * w
        r = vec_add_scaled(r, w, -alpha)

        r_norm_sq_new = dot_product(r, r)

        # Critério de parada
        epsilon = abs(r_norm_sq_new - r_norm_sq_old)
        erro_final = epsilon

        if epsilon < tol and k > 0:
            print(f"Convergência: Iter {iteracoes}, Epsilon {epsilon:.2e}")
            break

        # z = H.T @ r
        z = mat_T_vec_mul(H, r)

        z_norm_sq_new = dot_product(z, z)
        beta = z_norm_sq_new / z_norm_sq_old

        # p = z + beta * p
        p = vec_add_scaled(z, p, beta)

        z_norm_sq_old = z_norm_sq_new
        r_norm_sq_old = r_norm_sq_new

    end_time = time.time()

    return {
        "imagem_f": f,
        "iteracoes": iteracoes,
        "tempo_s": end_time - start_time,
        "erro_final": erro_final
    }

@app.route('/reconstruir', methods=['POST'])
def api_reconstruir():
    if not request.json:
        return jsonify({"status": "erro"}), 400

    config = request.json
    try:
        print(f"Lendo arquivos (Lento em Pure Python)...")
        H = ler_csv_como_matriz(config['caminho_h'])

        g_matriz = ler_csv_como_matriz(config['caminho_g'])
        g_raw = achatar_lista(g_matriz)

        S = int(config['s'])
        N = int(config['n'])

        g = []
        for l in range(S):
            gamma = math.sqrt(100 + (l**2)/20)
            for c in range(N):
                idx = l * N + c
                if idx < len(g_raw):
                    g.append(g_raw[idx] * gamma)
                else:
                    g.append(0.0)

        res = cgnr_pure(H, g)

        nome_saida = f"pure_py_{config.get('nome_arquivo_base')}.pgm"

        salvar_pgm(res['imagem_f'], int(config['largura']), int(config['altura']), nome_saida)

        return jsonify({
            "status": "sucesso",
            "imagem_gerada": nome_saida,
            "tempo_reconstrucao_s": res['tempo_s'],
            "iteracoes": res['iteracoes']
        })

    except Exception as e:
        print(f"Erro Pure Python: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)