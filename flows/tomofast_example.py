import os
import subprocess
import shutil

from onecode import (
    Logger,
    Project,
    slider,
    file_input,
    file_output,
    dropdown,
    number_input
)

from .utils import fortran_double_str
from .tomofast_vis import main


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
    out_path = os.path.join(data_path, "outputs")

    parameter_file = file_input('Parameter File', 'paramfile2.txt', types=[("TXT", ".txt")])

    # parameters to overwrite
    user_parameters = {
        'modelGrid.grav.file': file_input('Mesh file', 'model_grid2.txt', types=[("CSV", ".csv"), ("TXT", ".txt")]),
        'forward.data.grav.dataGridFile': file_input('Data file', 'data_grav.csv', types=[("CSV", ".csv"),("TXT", ".txt")]),
        'forward.matrixCompression.rate' : slider("Matrix Compression Rate", 0.15, min=0., max=1., step=0.01),
        'global.outputFolderPath ': out_path,                                                           # output in data folder so that files are automatically uploaded afterward
        'sensit.folderPath': os.path.join(out_path, 'SENSIT'),                                          # output in data folder so that files are automatically uploaded afterward
    }
    user_parameters['forward.data.grav.dataValuesFile'] = user_parameters['forward.data.grav.dataGridFile']

    # read default parameters
    with open(parameter_file) as f:
        lines = f.readlines()

    # Replace default parameters with user-selected parameters
    # => for each parameter to overwrite:
    #  - key is the parameter name in the parameter file, e.g. 'forward.matrixCompression.rate'
    #  - value is used to overwrite the default, e.g. 0.45
    new_params = []
    for line in lines:
        for key, value in user_parameters.items():
            if line.startswith(key):
                # if number, user fortran double-convention
                if isinstance(value, (int, float)):
                    value = fortran_double_str(value)

                line = f'{key} = {value} \n'
            elif line.startswith('modelGrid.size') :
                nx, ny, nz= line.split('= ')[1].split(' ')
        new_params.append(line)

    nx=int(nx)
    ny=int(ny)
    nz=int(nz)

    # write new parameter file
    new_param_file = parameter_file.replace('.txt', '_new.txt')
    with open(new_param_file, 'w') as f:
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
            "./tomofast-x/tomofastx", "-j", new_param_file
        ])
        returncode = result.returncode

    except:
        returncode = -1

    if returncode != -1:
        # make viz here (e.g. matplotlib)
        # instead of or in addition to plt.show() => plt.savefig(file_output(...))
        Logger.info("Tomofast ran successfully, generating graphical outputs")
        viz_folder = os.path.join(out_path, 'visualizations')
        os.makedirs(viz_folder, exist_ok=True)

        # ask user for dim and index to visualize
        #slice_dim = dropdown("[Viz] Dimension to visualize", "1", options=["0", "1", "2"])
        #slice_index = number_input("[Viz] Slice index to visualize", 30, min=1, step=1)
        main(
            user_parameters['modelGrid.grav.file'],
            os.path.join(out_path, 'model', 'grav_final_model_full.txt'),
            user_parameters['forward.data.grav.dataValuesFile'],
            os.path.join(out_path, 'data', 'grav_calc_final_data.txt'),
            slice_dim=int(0),
            slice_index=int(nx/2),
            draw_true_model=False,
            to_folder=viz_folder
        )
        main(
            user_parameters['modelGrid.grav.file'],
            os.path.join(out_path, 'model', 'grav_final_model_full.txt'),
            user_parameters['forward.data.grav.dataValuesFile'],
            os.path.join(out_path, 'data', 'grav_calc_final_data.txt'),
            slice_dim=int(1),
            slice_index=int(ny/2),
            draw_true_model=False,
            to_folder=viz_folder
        )
        main(
            user_parameters['modelGrid.grav.file'],
            os.path.join(out_path, 'model', 'grav_final_model_full.txt'),
            user_parameters['forward.data.grav.dataValuesFile'],
            os.path.join(out_path, 'data', 'grav_calc_final_data.txt'),
            slice_dim=int(2),
            slice_index=int(nz/2),
            draw_true_model=False,
            to_folder=viz_folder
        )

    # Here, expose all output files by recursively looping in the output directory
    # Alternatively, explicit declaration
    # e.g. file_output('file 1', os.path.join(out_path, 'model', 'grav_final_model_full.txt'))

    for dirpath, _, filenames in os.walk(out_path):
        for filename in filenames:
            # add if-statement here to filter files and not expose some of them,
            # e.g. if filename.endswith('.txt')

            suffix_list= tuple(['.txt','.jpg','.vtk'])
            if filename.endswith(suffix_list):
                f = os.path.join(dirpath, filename)
                file_output(filename, f)

    if returncode != 0:
        Logger.critical("An error occured while running Tomofast-x")
        raise RuntimeError(f"An error occured while running Tomofast-x [{returncode}]")
