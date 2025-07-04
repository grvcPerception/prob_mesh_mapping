## Raw MonteCarlo estimation values 

This folder contains the raw MonteCarlo estimation values that were used to obtain the polynomials to approximate the probabilistic quadric in a virtual reference frame where the plane points at [1;0;0]. Please refer to Figure 4 of the paper when checking this folder.

The folder consists of three files:
- `grid_values.csv` contains each pair of values Sigma_1_1 and Sigma_2_2 where the MonteCarlo estimation was performed.
- `q_virtual_11.csv` contains the values Q_1_1 for the correspondent Sigma_1_1 and Sigma_2_2 values in `grid_values.csv`. Note that Q_2_2(Sigma_1_1, Sigma_2_2) = Q_1_1(Sigma_2_2, Sigma_1_1), so the same file can be used for both.
- `q_virtual_33.csv` contains the values Q_3_3 for the correspondent Sigma_1_1 and Sigma_2_2 values in `grid_values.csv`.

If you just want to use our estimated polynomials, they are located at `code/math.h`.