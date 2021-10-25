# Install C++/CUDA extensions
for ext in mesh2sdf_cuda sol_nglod; do
    cd $ext && python3.8 setup.py clean --all install --user && cd -
done
