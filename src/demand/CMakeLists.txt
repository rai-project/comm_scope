set(comm_SOURCES ${comm_SOURCES}

PARENT_SCOPE)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cu
${CMAKE_CURRENT_LIST_DIR}/gpu_to_host_mt.cu
${CMAKE_CURRENT_LIST_DIR}/gpu_to_host.cu
${CMAKE_CURRENT_LIST_DIR}/host_to_gpu.cu
PARENT_SCOPE)