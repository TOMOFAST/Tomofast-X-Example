import os
import subprocess
import shutil

from onecode import (
    Logger,
    Project,
    slider,
    number_input,
    file_input,
    dropdown,
    checkbox,
    text_input,
    file_output
)

from .parameters import write_params


def run():
    # Select number of cores to use with MPI
    n_cores = slider(
        "Number of cores (0 for all available)",
        0,
        min=0,
        max=16,
        step=1
    )

    max_cores = os.cpu_count()
    if n_cores == 0 or n_cores > max_cores:
        n_cores = max_cores

    param_file = file_input('Parameter File', 'Parfile_mansf_slice.txt', types=[("TXT", ".txt")])

    # Input parameters and output folder definition
    Logger.info("Preparing parameter file...")
    data_path = Project().data_root
    param_file = os.path.join(data_path, 'parameters.txt')
    out_path = os.path.join(data_path, "outputs")

    nx = 2 # number_input('Nx', 2, min=0, step=1)
    ny = 128 # number_input('Ny', 128, min=0, step=1)
    nz = 32 # number_input('Nz', 32, min=0, step=1)

    grav_file = file_input('Model Grav. file', 'gravmag/mansf_slice/true_model_grav_3litho.txt', types=[("TXT", ".txt")])
    n_data = 256 # number_input('Grav. N data', 256, min=0, step=1)
    
    grid_file = file_input('Model Grid file', 'gravmag/mansf_slice/data_grid.txt', types=[("TXT", ".txt")])

    depth_type = "1" # dropdown("Depth Weighting Type", "1", options=["0", "1"])
    grav_power = 2 #number_input("Depth Weighting Grav. Power", 2, min=1, step=1)

    matrix_comp_type = "1-wavelet" # dropdown("Matrix Compression Type", "1-wavelet", options=["0-none", "1-wavelet"])
    matrix_comp_type = matrix_comp_type.split("-")[0]

    matrix_comp_rate = slider("Matrix Compression Rate", 0.15, min=0., max=1., step=0.01)

    prior_model_type = "1" # dropdown("Prior Model Type", "1", options=["0", "1"])
    prior_model_grav = 0. # number_input("Prior Model Grav.", 0.,)

    start_model_type = "1" # dropdown("Start Model Type", "1", options=["0", "1"])
    start_model_grav = 0. # number_input("Start Model Grav.", 0.,)

    n_major_itr = 60 # number_input("N Major Iterations", 60, min=1, step=1)
    n_minor_itr = 100 # number_input("N Minor Iterations", 100, min=1, step=1)
    write_model_itr = 0 # number_input("Write Model every N iteration", 0, min=0, step=1)
    min_residual = 1e-13 # number_input("Min. Residual", 1e-13, min=1e-13, step=1e-13)

    damping_grav_weight = 0 # slider("Damping Grav. Weight", 0., min=0., max=1., step=0.01)
    damping_norm_power = 2 # number_input("Damping Norm Power", 2, min=1, step=1)
    
    joint_grav_weight = 1 #slider("Inversion Grav. Problem Weight", 1., min=0., max=1., step=0.01)
    joint_magn_weight = 0 # slider("Inversion Magn. Problem Weight", 0., min=0., max=1., step=0.01)

    admm_enable = True # checkbox("Enable ADMM", True)
    admm_n_lithos = 0 # number_input("ADMM N Lithologies", 3, min=1, step=1)
    admm_grav_bounds = "-20. 20. 90. 130. 220. 260." # text_input("ADMM Grav. bounds (whitespace separated)", "-20. 20. 90. 130. 220. 260.")
    admm_grav_bounds = admm_grav_bounds.split(" ")
    admm_grav_weight = 1e-5 # slider("ADMM Grav. Weight", 1e-5, min=0., max=1., step=1e-5)

    # logic here to read the existing parameter file
    with open(param_file) as f:
        lines = f.readlines()

    new_params = []
    for line in lines:
        if line.startswith('forward.matrixCompression.rate'):
            line = f'forward.matrixCompression.rate = {matrix_comp_rate} \n'
        elif line.startswith('modelGrid.grav.file'):
            line = f'modelGrid.grav.file = {grav_file} \n'
        elif line.startswith('forward.data.grav.dataGridFile'):
            line = f'forward.data.grav.dataGridFile = {grid_file} \n'

        new_params.append(line)

    with open(param_file, 'w') as f:
        for l in new_params:
            f.write(l)


    if shutil.which('mpirun') is None:
        Logger.critical("MPI is not present on the machine => aborting")
        raise RuntimeError("MPI is not present on the machine!")

    # Run tomofast
    try:
        Logger.info(f"MPI available => Running Tomofast-x with {n_cores} cores")

        result = subprocess.run([
            "mpirun",
            "--oversubscribe",
            "-np", str(n_cores),
            "./tomofast-x/tomofastx", "-j", param_file
        ])
        returncode = result.returncode

    except:
        returncode = -1

    # Here, expose all output files by recursively looping in the output directory
    # Alternatively, explicit declaration
    # e.g. file_output('file 1', os.path.join(out_path, 'model', 'grav_final_model_full.txt'))

    for dirpath, _, filenames in os.walk(out_path):
        for filename in filenames:
            # add if-statement here to filter files and not expose some of them,
            # e.g. if filename.endswith('.txt')

            f = os.path.join(dirpath, filename)
            file_output(filename, f)

    if returncode != 0:
        Logger.critical("An error occured while running Tomofast-x")
        raise RuntimeError(f"An error occured while running Tomofast-x [{returncode}]")
