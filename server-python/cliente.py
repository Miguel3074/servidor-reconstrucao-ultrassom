import requests
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt

# ==============================================================================
# FUNÇÕES DE APOIO PARA O CLIENTE
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

def salvar_imagem(vetor_f, largura, altura, nome_arquivo="imagem_reconstruida.png"):
    """
    Converte o vetor da imagem 'f' em uma matriz 2D e a salva como um arquivo de imagem.
    """
    if len(vetor_f) != largura * altura:
        raise ValueError("O tamanho do vetor 'f' não corresponde às dimensões da imagem.")
    
    imagem_matrix = np.array(vetor_f).reshape((altura, largura))

    # Normaliza a imagem para melhor contraste visual
    f_min, f_max = imagem_matrix.min(), imagem_matrix.max()
    imagem_normalizada = (imagem_matrix - f_min) / (f_max - f_min)

    plt.imsave(nome_arquivo, imagem_normalizada, cmap='gray')
    print(f"  -> Imagem salva com sucesso como '{nome_arquivo}'")

# ==============================================================================
# LÓGICA PRINCIPAL DO CLIENTE
# ==============================================================================

# --- Configurações ---
URL_SERVIDOR_PYTHON = "http://127.0.0.1:5000/reconstruir"
# URL_SERVIDOR_JAVA = "http://127.0.0.1:8080/reconstruir" # (Quando você criar o servidor Java)

# Lista de sinais a serem enviados para reconstrução
SINAIS_DE_TESTE = {
    "imagem_1_60x60": r'Img1/G-1.csv',
    "imagem_2_60x60": r'Dados/Img1/G-2.csv', # Verifique o caminho correto
    "imagem_3_60x60": r'Dados/Img1/G-3.csv'  # Verifique o caminho correto
}

def main():
    print("--- Iniciando Aplicação Cliente ---")
    
    relatorio_final = []

    # Itera sobre a lista de sinais de teste
    for nome_amigavel, caminho_sinal_g in SINAIS_DE_TESTE.items():
        print(f"\n--------------------------------------------------")
        print(f"Processando: {nome_amigavel}")
        
        try:
            # 1. Carrega o vetor de sinal do arquivo
            g_vetor = carregar_ou_criar_npy(caminho_sinal_g, delimiter=',')
            
            # 2. Prepara o payload JSON para enviar ao servidor
            payload = {"sinal_g": g_vetor.tolist()} 
            
            # 3. Envia a requisição POST para o servidor e mede o tempo
            print(f"Enviando requisição para {URL_SERVIDOR_PYTHON}...")
            start_req = time.time()
            response = requests.post(URL_SERVIDOR_PYTHON, json=payload, timeout=300) # Timeout de 5min
            end_req = time.time()
            response.raise_for_status() # Lança um erro se a resposta for 4xx ou 5xx
            
            print(f"Resposta recebida em {end_req - start_req:.2f} segundos.")
            resultado = response.json()
            
            # 4. Processa a resposta e salva os resultados
            metadata = resultado['metadata']
            
            # Adiciona o nome do arquivo ao relatório
            metadata['arquivo_sinal'] = caminho_sinal_g
            relatorio_final.append(metadata)

            # Salva a imagem recebida
            nome_arquivo_saida = f"resultado_{nome_amigavel}_py.png"
            salvar_imagem(
                resultado['imagem_f'], 
                metadata['pixels_largura'], 
                metadata['pixels_altura'], 
                nome_arquivo_saida
            )

            # 5. Aguarda um tempo aleatório antes da próxima requisição
            intervalo = random.uniform(2, 6) # Espera entre 2 e 6 segundos
            print(f"Aguardando {intervalo:.2f} segundos...")
            time.sleep(intervalo)

        except FileNotFoundError:
            print(f"AVISO: Arquivo de sinal não encontrado: {caminho_sinal_g}. Pulando para o próximo.")
            continue
        except requests.exceptions.RequestException as e:
            print(f"ERRO CRÍTICO: Não foi possível se comunicar com o servidor. {e}")
            print("Verifique se o servidor_python.py está em execução.")
            break
            
    print("\n======================================================")
    print("RELATÓRIO FINAL CONSOLIDADO")
    print("======================================================")
    if not relatorio_final:
        print("Nenhuma imagem foi processada.")
    else:
        for item in relatorio_final:
            print(f"\n- Sinal de Entrada: {item['arquivo_sinal']}")
            print(f"  - Algoritmo: {item['algoritmo']}")
            print(f"  - Duração (servidor): {item['tempo_s']:.4f}s")
            print(f"  - Iterações: {item['iteracoes']}")
            print(f"  - Início: {item['inicio_reconstrucao']}")
            print(f"  - Término: {item['termino_reconstrucao']}")

if __name__ == '__main__':
    main()