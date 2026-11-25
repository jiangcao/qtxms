# source envs/daint/mn_env.sh   (inside a job started with --uenv=prgenv-gnu/24.11:v1, after activate.sh; $R = repo)
# Several nodes: cray-mpich of the uenv under mpi4py, NCCL over Slingshot through aws-ofi-nccl of the uenv. MPI carries
# host data only (dist.py): no GPU-aware MPI.
# mpi4py wheel: needs a library NAMED libmpi.so.12 (MPI4PY_LIBMPI alone fails: the cray library is libmpi_gnu_123.so.12)
# -> a link of that name in .tools/mpi_shim. If the wheel does not run on cray-mpich (MPICH 3.4 ABI), mpi4py is built
# from source against it into .tools/mpi4py_cray (mn_build_mpi4py).
_l=${UENV_MOUNT_POINT:-/user-environment}/env/default/lib
mkdir -p $R/.tools/mpi_shim
ln -sfn $_l/libmpi_gnu_123.so.12 $R/.tools/mpi_shim/libmpi.so.12
export LD_LIBRARY_PATH=$R/.tools/mpi_shim:$_l${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MPI4PY_MPIABI=mpich MPI4PY_LIBMPI=$R/.tools/mpi_shim/libmpi.so.12      # else the wheel takes the MPICH of the venv (rpath)
export MPICH_GPU_SUPPORT_ENABLED=0 QTX_LAUNCH=srun
if ldd $_l/libmpi_gnu_123.so.12 $_l/libnccl-net.so 2>/dev/null | grep -q "libfabric.*not found"; then
  _f=$(ls -d /opt/cray/libfabric/[0-9]*/lib64 2>/dev/null | sort -V | tail -1); export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$_f
fi
# NCCL / libfabric cxi settings of https://docs.cscs.ch/software/communication/nccl/ ; FI_CXI_RDZV_*: its workaround for hangs
export NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 NCCL_PROTO=^LL128 NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software FI_MR_CACHE_MONITOR=userfaultfd NCCL_NCHANNELS_PER_NET_PEER=4
export FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0
echo "mn_env: libraries not found by cray-mpich / NCCL plugin: $(ldd $_l/libmpi_gnu_123.so.12 $_l/libnccl-net.so 2>/dev/null | grep 'not found' | sort -u | tr '\n' ' ')"
ldd $_l/libmpi_gnu_123.so.12 $_l/libnccl-net.so 2>/dev/null | grep -E "libfabric|libcxi" | sort -u
mn_build_mpi4py() {  # on a compute node inside the uenv
  local b=${UENV_MOUNT_POINT:-/user-environment}/env/default/bin
  UV_CACHE_DIR=$R/.tools/uv-cache UV_LINK_MODE=copy PATH=$b:$PATH MPICC=$b/mpicc $R/.tools/uv-venv/bin/uv pip install \
    --python $R/.venv-daint/bin/python --no-binary mpi4py --target $R/.tools/mpi4py_cray mpi4py==4.1.2
}
if [ -d $R/.tools/mpi4py_cray/mpi4py ]; then export PYTHONPATH=$R/.tools/mpi4py_cray:$PYTHONPATH; unset MPI4PY_LIBMPI MPI4PY_MPIABI; fi
unset _l _f
