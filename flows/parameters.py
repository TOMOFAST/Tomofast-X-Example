import os
from typing import List

from .utils import fortran_double_str


def write_params(
    to_file: str,
    global_output_path: str,
    nx: int,
    ny: int,
    nz: int,
    model_grav_file: str,
    forward_n_data: int,
    forward_data_grid_file: str,
    forward_depth_type: int,
    forward_grav_power: int,
    forward_matrix_comp_type: int,
    forward_matrix_comp_rate: float,
    inv_prior_model_type: int,
    inv_prior_model_grav: float,
    inv_start_model_type: int,
    inv_start_model_grav: float,
    inv_n_major_itr: int,
    inv_n_minor_itr: int,
    inv_write_model_itr: int,
    inv_min_residual: float,
    inv_damping_grav_weight: float,
    inv_damping_norm_power: int,
    inv_joint_grav_weight: float,
    inv_joint_magn_weight: float,
    inv_admm_enable: bool,
    inv_admm_n_lithos: int,
    inv_admm_grav_bounds: List[float],
    inv_admm_grav_weight: float
):
    param_content = f"""
===================================================================================
GLOBAL
===================================================================================
global.outputFolderPath             = {global_output_path}
global.description                  = Gravity inversion with ADMM constraints

===================================================================================
MODEL GRID parameters
===================================================================================
# nx ny nz
modelGrid.size                      = {nx} {ny} {nz}
modelGrid.grav.file                 = {model_grav_file}

===================================================================================
DATA parameters
===================================================================================
forward.data.grav.nData             = {forward_n_data}
forward.data.grav.dataGridFile      = {forward_data_grid_file}
forward.data.grav.dataValuesFile    = {os.path.join(global_output_path, 'data', 'grav_calc_read_data.txt')}

===================================================================================
DEPTH WEIGHTING
===================================================================================
forward.depthWeighting.type         = {forward_depth_type}
forward.depthWeighting.grav.power   = {fortran_double_str(forward_grav_power)}

===================================================================================
SENSITIVITY KERNEL
===================================================================================
sensit.readFromFiles                = 0
sensit.folderPath                   = {os.path.join(global_output_path, 'SENSIT')}

===================================================================================
MATRIX COMPRESSION
===================================================================================
# 0-none, 1-wavelet compression.
forward.matrixCompression.type      = {forward_matrix_comp_type}
forward.matrixCompression.rate      = {forward_matrix_comp_rate}

===================================================================================
PRIOR MODEL
===================================================================================
inversion.priorModel.type           = {inv_prior_model_type}
inversion.priorModel.grav.value     = {fortran_double_str(inv_prior_model_grav)}

===================================================================================
STARTING MODEL
===================================================================================
inversion.startingModel.type        = {inv_start_model_type}
inversion.startingModel.grav.value  = {fortran_double_str(inv_start_model_grav)}

===================================================================================
INVERSION parameters
===================================================================================
inversion.nMajorIterations          = {inv_n_major_itr}
inversion.nMinorIterations          = {inv_n_minor_itr}
inversion.writeModelEveryNiter      = {inv_write_model_itr}
inversion.minResidual               = {fortran_double_str(inv_min_residual)}

===================================================================================
MODEL DAMPING (m - m_prior)
===================================================================================
inversion.modelDamping.grav.weight  = {fortran_double_str(inv_damping_grav_weight)}
inversion.modelDamping.normPower    = {fortran_double_str(inv_damping_norm_power)}

===================================================================================
JOINT INVERSION parameters
===================================================================================
inversion.joint.grav.problemWeight  = {fortran_double_str(inv_joint_grav_weight)}
inversion.joint.magn.problemWeight  = {fortran_double_str(inv_joint_magn_weight)}

===================================================================================
ADMM constraints
===================================================================================
inversion.admm.enableADMM           = {'1' if inv_admm_enable else '0'}
inversion.admm.nLithologies         = {inv_admm_n_lithos}
inversion.admm.grav.bounds          = {' '.join(inv_admm_grav_bounds)}
inversion.admm.grav.weight          = {fortran_double_str(inv_admm_grav_weight)}

===================================================================================

"""
    with open(to_file, 'w') as f:
        f.write(param_content)
