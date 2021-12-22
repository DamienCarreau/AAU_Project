"""
Creation of the alternative dataset.

Create an averaged and a not averaged set of dataset. Use informations in "trajectory" directory
which have been generated using "post_processing_generator".

n : number of values used to average
distance_exclusion_value : if the distance of the smooth trajectory to the MPC trajectory is above this value, trajectory is not accepted in the dataset

nb_dataset : number of datasets (defined in post_processing_generator)
nb_trajectory : number of trajectories (defined in post_processing_generator)
DT: time tick (defined in post_processing_generator)
"""

import post_processing_tools
import statistics
import post_processing_generator

# generration parameters, defined in post_processing_generator
nb_dataset = post_processing_generator.nb_dataset # number of dataset
nb_trajectory = post_processing_generator.nb_trajectory # number of trajectories
DT = post_processing_generator.DT # time tick

# averaging
n = 5 # number of values used to average
distance_exclusion_value = 0.3 #  if the distance of the smooth trajectory to the MPC trajectory is above this value, trajectory is not accepted in the dataset

def main():
    for i in range(0,nb_dataset):
        print('averaging dataset ' + str(i))

        # load files generated in post_processing_generator
        array_trajectory, a_mpc, d_mpc = post_processing_tools.load_array_trajectory_state(f'./post_processing_data/trajectory/arrayTrajectoriesStates{i}.npy')
        x0 = post_processing_tools.load_npy_file(f'./post_processing_data/trajectory/x0_array{i}_data.npy')

        # post processing with a given n
        mean_coeff = int(n / 2) - 1
        a_smooth, d_smooth = post_processing_tools.post_process_datas(a_mpc, d_mpc, mean_coeff)

        # load data to modify
        data_to_modify = post_processing_tools.load_npy_file(f'./post_processing_data/trajectory/training{i}_data.npy')

        # calcul of distance distance between new dataset and not post processed dataset
        _, _, distance_mpc_array = post_processing_tools.evaluate_datas(x0, a_smooth, a_mpc, d_smooth, d_mpc, DT, nb_trajectory)

        # create arrays of states (inputs and outputs of the neural networks) + remove outlier trajectories
        pp_array_state, array_state, pp_data, nb_excluded = post_processing_tools.build_array_state(array_trajectory, data_to_modify,
                                                                                                                     a_smooth, d_smooth,
                                                                                                                     distance_mpc_array,
                                                                                                                     distance_exclusion_value,
                                                                                                                     nb_trajectory)
        # save post processed dataset
        post_processing_tools.save_npy_file(f'./post_processing_data/ppTraining/ppTraining{i}.npy', pp_array_state)
        post_processing_tools.save_npy_file(f'./post_processing_data/ppTraining/ppTraining{i}_data.npy', pp_data)

        # save without post processing dataset
        post_processing_tools.save_npy_file(f'./post_processing_data/training/training{i}.npy', array_state)
        post_processing_tools.save_npy_file(f'./post_processing_data/training/training{i}_data.npy', pp_data)
main()