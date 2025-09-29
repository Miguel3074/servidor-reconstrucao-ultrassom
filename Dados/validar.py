import numpy as np
import os

def carregar_dados_csv(caminho_csv, delimiter=';'):
    if not os.path.exists(caminho_csv):
        print(f"ERRO: Arquivo não encontrado em '{caminho_csv}'")
        return None
    print(f"Carregando arquivo de teste: {caminho_csv}")
    return np.loadtxt(caminho_csv, delimiter=delimiter)

print("--- Iniciando a Validação das Operações Básicas ---")

M = carregar_dados_csv('M.csv')
N = carregar_dados_csv('N.csv')
vetor_a = carregar_dados_csv('a.csv') 
resultado_esperado_MN = carregar_dados_csv('MN.csv')
resultado_esperado_Ma = carregar_dados_csv('aM.csv')

if any(x is None for x in [M, N, vetor_a, resultado_esperado_MN, resultado_esperado_Ma]):
    print("\nUm ou mais arquivos não foram encontrados. Verifique os caminhos e nomes.")
else:
    print("\nCalculando operações com NumPy...")
    
    resultado_calculado_MN = M @ N
    resultado_calculado_aM = vetor_a @ M

    print("Cálculos finalizados. Iniciando comparação...")

    print("\n--- Resultados da Validação ---")

    if np.allclose(resultado_calculado_MN, resultado_esperado_MN):
        print("✅ SUCESSO: O resultado de M * N está CORRETO.")
    else:
        print("❌ FALHA: O resultado de M * N está INCORRETO.")

    if np.allclose(np.round(resultado_calculado_aM, 2), resultado_esperado_Ma):
        print("✅ SUCESSO: O resultado de a * M (vetor) está CORRETO.")
    else:
        print("❌ FALHA: O resultado de a * M (vetor) está INCORRETO.")