import matplotlib.pyplot as plt
import glob
import os
import json

arquivos_pgm = glob.glob("*.pgm")

print(f"Encontrados {len(arquivos_pgm)} arquivos PGM para processar.")

for pgm_path in arquivos_pgm:
    try:
        json_path = pgm_path.replace('.pgm', '.json')
        meta = {}

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
        else:
            print(f"Aviso: JSON não encontrado para {pgm_path}. Usando dados genéricos.")

        img = plt.imread(pgm_path)

        plt.figure(figsize=(6, 7))
        plt.imshow(img, cmap='gray')

        # Título
        algo = meta.get('algo', 'CGNR (C++)')
        nome = meta.get('nome_base', 'Desconhecido')
        dim = f"{meta.get('largura','?')}x{meta.get('altura','?')}"
        plt.title(f"Algoritmo: {algo}\nImg: {nome} ({dim})", fontsize=12, fontweight='bold')

        texto_inferior = (
            f"Inicio: {meta.get('inicio', '?')}\n"
            f"Fim:    {meta.get('fim', '?')}\n"
            f"Iteracoes: {meta.get('iteracoes', 0)} | Tempo: {meta.get('tempo_s', 0):.4f}s\n"
            f"Erro Final: {meta.get('erro_final', 0):.2e}"
        )

        plt.xlabel(texto_inferior, fontsize=10, bbox=dict(facecolor='white', alpha=0.9))

        plt.xticks([])
        plt.yticks([])

        nome_png = pgm_path.replace('.pgm', '_FINAL.png')
        plt.tight_layout()
        plt.savefig(nome_png, dpi=100)
        plt.close()

        print(f"Gerado com sucesso: {nome_png}")

    except Exception as e:
        print(f"Erro ao processar {pgm_path}: {e}")

print("Conversão concluída!")