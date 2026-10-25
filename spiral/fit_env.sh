# source me: common env for MPS spiral fits on a 1000-slice band dataset
# (run from villa/spiral-fitting; paths are relative to this repo's root)
_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
export FIT_SPIRAL_DEVICE=${FIT_SPIRAL_DEVICE:-mps}
export FIT_SPIRAL_CACHE_DIR=$_ROOT/cache/spiral
export FIT_SPIRAL_OUT_DIR=$_ROOT/spiral/runs
BASE='"z_begin":600,"z_end":1600,"input_use_tracks":true,"input_use_normals":false,"input_use_gradient_magnitude":false,"input_use_winding_inference":false,"input_use_outer_shell":false,"input_use_fibers":false,"input_use_pcl_absolute":false,"input_use_pcl_relative":false,"input_use_pcl_same_winding":false,"input_use_pcl_drawn_control_points":false,"dense_spacing_mode":"grad_mag","loss_weight_dense_spacing":0,"loss_weight_shell_outer":0'
