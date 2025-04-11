generate_direct_only: 给定起点，只生成直线。这个文件似乎没用到

generate_curve_direct: 这个会生成一个直线(straight_line_trajectory.csv)，和对应的曲线文件(smooth_multi_point_traj.csv)

generate_short_plot.py: 这个会生成一个自行车约束轨迹，我们最后不会用到其实


sim_compare_Q.py: Compare the performance of different Q
sim_compare_R.py: Compare the performance of different R

sim_origin_regulator_MPC: 终点在原点，不需要导入其他trajectory
sim_one_straight_offset: 终点不在原点。此时没有trajectory文件导入，而是在sim文件内部生成
sim_combination_straight_lines: 使用多条直线组合而成的轨迹，使用文件为straight_line_trajectory.csv

sim_smooth_fixed_ref: 使用smooth_multi_point_traj.csv的曲线轨迹，并且参考点固定且密集

