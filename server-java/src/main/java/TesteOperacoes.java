import org.apache.commons.math3.linear.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

// Arquivo salvo em: src/main/java/TesteOperacoes.java
// SEM declaração de package

public class TesteOperacoes {

    /**
     * Implementa o algoritmo CGNR para reconstrução de imagem.
     *
     */
    public static Map<String, Object> cgnr(RealMatrix H, RealVector g) {
        System.out.println("\n--- Iniciando Algoritmo CGNR em Java ---");
        long startTime = System.nanoTime(); // Medir o tempo total de execução

        int maxIter = 10;
        
        // Inicialização conforme o pseudocódigo
        RealVector f = new ArrayRealVector(H.getColumnDimension()); // f0 = 0
        RealVector r = g.subtract(H.operate(f));
        RealVector z = H.transpose().operate(r);
        RealVector p = z.copy();

        double z_dot_z_old = z.dotProduct(z);
        
        int iterationsDone = 0;
        for (int i = 0; i < maxIter; i++) {
            iterationsDone = i + 1;
            
            RealVector w = H.operate(p);
            double alpha = z_dot_z_old / w.dotProduct(w);
            
            f = f.add(p.mapMultiply(alpha));
            r = r.subtract(w.mapMultiply(alpha));
            z = H.transpose().operate(r);
            
            double z_dot_z_new = z.dotProduct(z);

            // O requisito principal é parar em 10 iterações.

            double beta = z_dot_z_new / z_dot_z_old;
            p = z.add(p.mapMultiply(beta));

            z_dot_z_old = z_dot_z_new;
        }

        long endTime = System.nanoTime();
        double duration = (endTime - startTime) / 1e9; // Convertendo para segundos

        System.out.printf("Execução em Java finalizada em %.6f segundos.\n", duration);

        Map<String, Object> results = new HashMap<>();
        results.put("imagem_f", f);
        results.put("iteracoes", iterationsDone);
        results.put("tempo_s", duration);
        
        return results;
    }

    // --- Exemplo de Uso ---
    public static void main(String[] args) {
        // Criando dados fictícios com dimensões menores para o teste rodar rápido
        int S = 100;
        int N = 20;
        
        // Criando matriz H e vetor g com valores aleatórios
        RealMatrix H_teste = MatrixUtils.createRealMatrix(S, N);
        RealVector g_teste = new ArrayRealVector(S);
        Random rand = new Random();
        for (int i = 0; i < S; i++) {
            g_teste.setEntry(i, rand.nextDouble());
            for (int j = 0; j < N; j++) {
                H_teste.setEntry(i, j, rand.nextDouble());
            }
        }
        
        Map<String, Object> resultado_java = cgnr(H_teste, g_teste);
        System.out.println("Iterações executadas: " + resultado_java.get("iteracoes"));
        // System.out.println("Vetor da imagem reconstruída (f): " + resultado_java.get("imagem_f"));
    }
}