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

    # Input parameters and output folder definition
    Logger.info("Preparing parameter file...")
    data_path = Project().data_root
    param_file = os.path.join(data_path, 'parameters.txt')
    out_path = os.path.join(data_path, "outputs")

    nx = number_input('Nx', 2, min=0, step=1)
    ny = number_input('Ny', 128, min=0, step=1)
    nz = number_input('Nz', 32, min=0, step=1)

    grav_file = file_input('Model Grav. file', 'gravmag/mansf_slice/true_model_grav_3litho.txt', types=[("TXT", ".txt")])
    n_data = number_input('Grav. N data', 256, min=0, step=1)
    
    grid_file = file_input('Model Grid file', 'gravmag/mansf_slice/data_grid.txt', types=[("TXT", ".txt")])

    depth_type = dropdown("Depth Weighting Type", "1", options=["0", "1"])
    grav_power = number_input("Depth Weighting Grav. Power", 2, min=1, step=1)

    matrix_comp_type = dropdown("Matrix Compression Type", "1-wavelet", options=["0-none", "1-wavelet"])
    matrix_comp_type = matrix_comp_type.split("-")[0]

    matrix_comp_rate = slider("Matrix Compression Rate", 0.15, min=0., max=1., step=0.01)

    prior_model_type = dropdown("Prior Model Type", "1", options=["0", "1"])
    prior_model_grav = number_input("Prior Model Grav.", 0.,)

    start_model_type = dropdown("Start Model Type", "1", options=["0", "1"])
    start_model_grav = number_input("Start Model Grav.", 0.,)

    n_major_itr = number_input("N Major Iterations", 60, min=1, step=1)
    n_minor_itr = number_input("N Minor Iterations", 100, min=1, step=1)
    write_model_itr = number_input("Write Model every N iteration", 0, min=0, step=1)
    min_residual = number_input("Min. Residual", 1e-13, min=1e-13, step=1e-13)

    damping_grav_weight = slider("Damping Grav. Weight", 0., min=0., max=1., step=0.01)
    damping_norm_power = number_input("Damping Norm Power", 2, min=1, step=1)
    
    joint_grav_weight = slider("Inversion Grav. Problem Weight", 1., min=0., max=1., step=0.01)
    joint_magn_weight = slider("Inversion Magn. Problem Weight", 0., min=0., max=1., step=0.01)

    admm_enable = checkbox("Enable ADMM", True)
    admm_n_lithos = number_input("ADMM N Lithologies", 3, min=1, step=1)
    admm_grav_bounds = text_input("ADMM Grav. bounds (whitespace separated)", "-20. 20. 90. 130. 220. 260.")
    admm_grav_bounds = admm_grav_bounds.split(" ")
    admm_grav_weight = slider("ADMM Grav. Weight", 1e-5, min=0., max=1., step=1e-5)

    # Write parameters to file
    write_params(
        param_file,
        out_path,
        nx,
        ny,
        nz,
        grav_file,
        n_data,
        grid_file,
        depth_type,
        grav_power,
        matrix_comp_type,
        matrix_comp_rate,
        prior_model_type,
        prior_model_grav,
        start_model_type,
        start_model_grav,
        n_major_itr,
        n_minor_itr,
        write_model_itr,
        min_residual,
        damping_grav_weight,
        damping_norm_power,
        joint_grav_weight,
        joint_magn_weight,
        admm_enable,
        admm_n_lithos,
        admm_grav_bounds,
        admm_grav_weight
    )

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
            file_output(f, f)

    if returncode != 0:
        Logger.critical("An error occured while running Tomofast-x")
        raise RuntimeError(f"An error occured while running Tomofast-x [{returncode}]")
