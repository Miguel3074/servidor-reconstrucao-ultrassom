import requests
import time
import random
import json
import csv
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_CPP = "http://127.0.0.1:5000/reconstruir"
URL_PYTHON = "http://127.0.0.1:5001/reconstruir"

NUMERO_DE_TESTES = 10
MAX_THREADS = 4
SEED = 42

OPCOES = [
    {"H": "dados/modelo1/H-1.csv", "G": ["dados/modelo1/G-1.csv"], "w": 60, "h": 60, "s": 794, "n": 64, "tipo": "M1"},
    {"H": "dados/modelo1/H-1.csv", "G": ["dados/modelo1/G-2.csv"], "w": 60, "h": 60, "s": 794, "n": 64, "tipo": "M1"},
    {"H": "dados/modelo1/H-1.csv", "G": ["dados/modelo1/G-3.csv"], "w": 60, "h": 60, "s": 794, "n": 64, "tipo": "M1"},
    {"H": "dados/modelo2/H-2.csv", "G": ["dados/modelo2/G-1.csv"], "w": 30, "h": 30, "s": 436, "n": 64, "tipo": "M2"},
    {"H": "dados/modelo2/H-2.csv", "G": ["dados/modelo2/G-2.csv"], "w": 30, "h": 30, "s": 436, "n": 64, "tipo": "M2"},
    {"H": "dados/modelo2/H-2.csv", "G": ["dados/modelo2/G-3.csv"], "w": 30, "h": 30, "s": 436, "n": 64, "tipo": "M2"}
]

def gerar_tarefas_aleatorias(qtd):
    random.seed(SEED)
    tarefas = []

    print(f"\n--- Sorteando {qtd} tarefas aleatórias ---")
    for i in range(qtd):
        config_base = random.choice(OPCOES)
        g_escolhido = random.choice(config_base["G"])

        id_unico = str(uuid.uuid4())[:8]

        tarefa = {
            "id": i + 1,
            "caminho_h": config_base["H"],
            "caminho_g": g_escolhido,
            "nome_arquivo_base": f"teste_{i+1}_{config_base['tipo']}_{id_unico}",
            "largura": config_base["w"],
            "altura": config_base["h"],
            "s": config_base["s"],
            "n": config_base["n"]
        }
        tarefas.append(tarefa)
    return tarefas

def enviar_uma_tarefa(dados_pacote):
    url, tarefa, nome_servidor = dados_pacote

    while True:
        try:
            start = time.time()
            resp = requests.post(url, json=tarefa)
            duracao_req = time.time() - start

            if resp.status_code == 200:
                dados = resp.json()
                img_nome = dados.get('imagem_gerada') or dados.get('imagem_gerada_ruidosa')
                iters = dados.get('iteracoes') or dados.get('iteracoes_executadas')
                tempo_algo = dados.get('tempo_reconstrucao_s', 0.0)
                memoria = dados.get('memoria_mb', 0.0)

                print(f"[{nome_servidor}] CONCLUÍDO: {tarefa['nome_arquivo_base']} ({tempo_algo:.4f}s)")

                return {
                    "tarefa": tarefa["nome_arquivo_base"],
                    "versao": nome_servidor,
                    "status": "sucesso",
                    "iteracoes": iters,
                    "tempo_algoritmo_s": tempo_algo,
                    "tempo_total_req_s": duracao_req,
                    "memoria_mb": memoria,
                    "imagem": img_nome,
                    "erro_msg": ""
                }

            elif resp.status_code == 503:
                wait_time = random.uniform(4.0, 10.0)
                print(f"[{nome_servidor}] Servidor CHEIO. Aguardando {wait_time:.1f}s p/ {tarefa['nome_arquivo_base']}...")
                time.sleep(wait_time)
                continue

            else:
                print(f"[{nome_servidor}] ERRO {resp.status_code} em {tarefa['nome_arquivo_base']}")
                return {
                    "tarefa": tarefa["nome_arquivo_base"],
                    "versao": nome_servidor,
                    "status": "erro_http",
                    "iteracoes": 0, "tempo_algoritmo_s": 0.0, "tempo_total_req_s": 0.0, "memoria_mb": 0.0,
                    "imagem": "", "erro_msg": f"HTTP {resp.status_code}"
                }

        except Exception as e:
            print(f"[{nome_servidor}] FALHA em {tarefa['nome_arquivo_base']}: {e}")
            return {
                "tarefa": tarefa["nome_arquivo_base"],
                "versao": nome_servidor,
                "status": "falha_conexao",
                "iteracoes": 0, "tempo_algoritmo_s": 0.0, "tempo_total_req_s": 0.0, "memoria_mb": 0.0,
                "imagem": "", "erro_msg": str(e)
            }

def executar_lote_paralelo(nome_servidor, url, lista_tarefas):
    print(f"\n{'='*60}\n>>> INICIANDO TESTES PARALELOS: {nome_servidor} ({url})\n{'='*60}")
    resultados = []
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for tarefa in lista_tarefas:
            intervalo = random.uniform(3.0, 6.0)
            print(f"[{nome_servidor}] Enviando {tarefa['nome_arquivo_base']}... (Prox em {intervalo:.1f}s)")

            future = executor.submit(enviar_uma_tarefa, (url, tarefa, nome_servidor))
            futures.append(future)
            time.sleep(intervalo)

        print(f"\n[{nome_servidor}] Todas as tarefas enviadas! Aguardando retornos...\n")

        for future in as_completed(futures):
            res = future.result()
            resultados.append(res)

    return resultados

def salvar_relatorio_csv_formatado(res_py, res_cpp, nome_arquivo="relatorio_final.csv"):
    todos_dados = res_py + res_cpp
    colunas = ["tarefa", "versao", "status", "iteracoes", "tempo_algoritmo_s", "tempo_total_req_s", "memoria_mb", "imagem", "erro_msg"]

    arquivo_existe = os.path.isfile(nome_arquivo)

    try:
        with open(nome_arquivo, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=colunas, delimiter=';')

            if not arquivo_existe:
                writer.writeheader()

            for linha in todos_dados:
                linha_formatada = {}
                for k in colunas:
                    valor = linha.get(k, "")
                    if isinstance(valor, float):
                        valor = f"{valor:.4f}".replace('.', ',')
                    linha_formatada[k] = valor
                writer.writerow(linha_formatada)
        print(f"\n[SUCESSO] Relatório atualizado: {os.path.abspath(nome_arquivo)}")
    except Exception as e:
        print(f"\n[ERRO] CSV: {e}")

if __name__ == "__main__":
    fila = gerar_tarefas_aleatorias(NUMERO_DE_TESTES)

    #res_py = executar_lote_paralelo("Python", URL_PYTHON, fila)
    #res_cpp = []

    res_py = []
    res_cpp = executar_lote_paralelo("C++", URL_CPP, fila)

    salvar_relatorio_csv_formatado(res_py, res_cpp)