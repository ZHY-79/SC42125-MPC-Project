# MPC Project

Group members: Hongyu Zhou (6148123), Yuning Liu (6143237)

```generate_curve_direct.py```: Generate a straight line saved in the ```straight_line_trajectory.csv``` and a corresponding smooth trajectory in ```smooth_multi_point_traj.csv```

```sim_compare_Q.py```: Compare the performance of different Q. The results are saved in the folder ```compare_Q```
```sim_compare_R.py```: Compare the performance of different R. The results are saved in the folder ```compare_R```

```sim_origin_regulator.py```: The target is at the origin. There is no external trajectory file needed. 

```sim_one_straight_offset.py```: The target is not at the origin. There is no external trajectory file needed. 

```sim_combination_straight_lines.py```: The trajectory is a combination of straight lines generated in the ```straight_line_trajectory.csv```, with the reference point fixed at the intersection of different straight lines. 

```sim_smooth_fixed_ref.py```: The trajectory is the smooth one in the ```smooth_multi_point_traj.csv```, and the fixed reference points are dense. 