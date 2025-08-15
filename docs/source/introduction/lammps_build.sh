#!/bin/bash

# this script should be run while the Orchestrator environment is active in order to properly link and build with KIM-API

git clone --depth 1 --branch stable_22Jul2025 https://github.com/lammps/lammps.git lammps_stable_22Jul2025

module load mkl
export CC=mpicc
export CXX=mpic++

mkdir lammps_stable_22Jul2025/orchestrator_build

cat << 'EOF' > lammps_stable_22Jul2025/iap_packages.cmake
set(IAP_PACKAGES
  DIFFRACTION
  EXTRA-FIX
  KIM
  KSPACE
  MANYBODY
  MC
  MEAM
  MISC
  ML-HDNNP
  ML-IAP
  ML-PACE
  ML-SNAP
  MOLECULE
  PYTHON
  RIGID)

foreach(PKG ${IAP_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()
EOF

cd lammps_stable_22Jul2025/orchestrator_build
cmake -C ../iap_packages.cmake -D BUILD_MPI=yes -D BUILD_OMP=no -D LAMMPS_EXCEPTIONS=yes -D BUILD_SHARED_LIBS=yes ../cmake/
cmake --build . --parallel 8

# orchestrator environment needs to set PYTHONPATH and LD_LIBRARY_PATH variables. These can be set in the auto install script or in the source_me file at a later date:
# export PYTHONPATH=path/to/lammps_stable_22Jul2025/python:${PYTHONPATH}
# export LD_LIBRARY_PATH=/path/to/lammps_stable_22Jul2025/orchestrator_build:${LD_LIBRARY_PATH}
