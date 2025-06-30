1- Para compilar e executar k-means sequencial:
g++ kmeans.cpp -o kmeans -lstdc++ -lm
./kmeans


2- Para compilar e executar k-means paralelo:
g++ open_mp_cpu.cpp -o open_mp_cpu -lstdc++ -lm -fopenmp
./open_mp_cpu


3- Para compilar e executar k-means cuda:
sudo apt install nvidia-cuda-toolkit (instalar o compilador nvcc)
nvcc kmeans_cuda.cu -o kmeans_cuda
./kmeans_cuda

