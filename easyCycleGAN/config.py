from sketch_data_handler import *

class config(object):
    batch_size = 30
    target_size = 256
    learning_rate = 0.00007
    step = 1000000000
    lr_lower_bound = 0.000001
    lr_update_step = 1000000
    n_base = 20
    n_stack = 4
    n_z = 100
    gamma = 0.5
    log_step = 50

    summary_path = "./summary"

    pen_paths_file = "pen_list.txt"
    sketch_paths_file = "sketch_list.txt"

    pen_data_handler = SketchDataHandler(batch_size, target_size, pen_paths_file)
    sketch_data_handler = SketchDataHandler(batch_size, target_size, sketch_paths_file)

