cmake_minimum_required(VERSION 3.10)
project(ReconstrucaoImagem)

# Define o padrão C++ para C++17 (para std::filesystem, embora não usemos aqui)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Encontrar a biblioteca Eigen ---
# Tenta encontrar o Eigen automaticamente (funciona no Linux/Mac com 'brew install')
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Se você estiver no Windows e baixou o zip, descomente a linha abaixo
# e COLOQUE O CAMINHO CORRETO para a pasta que você descompactou (a pasta que contém 'Eigen/Dense')
# include_directories("C:/libs/eigen-3.4.0")

# Adiciona os "includes" do Eigen ao projeto
include_directories(${EIGEN3_INCLUDE_DIRS})

# --- Adicionar o executável ---
# Diz ao CMake para criar um programa chamado "reconstrucao" a partir de "main.cpp"
add_executable(reconstrucao main.cpp)

# Linka as bibliotecas do Eigen (necessário para alguns módulos)
target_link_libraries(reconstrucao PRIVATE Eigen3::Eigen)

# Habilita otimizações
set(CMAKE_BUILD_TYPE Release)