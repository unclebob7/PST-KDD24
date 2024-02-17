********************data downloading********************
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
[*********************100%***********************]  1 of 1 completed
Shape of DataFrame:  (84131, 8)
Successfully added technical indicators
Successfully added turbulence index
********************start training********************
============Start Ensemble Strategy============
============================================
turbulence_threshold:  160.33727482215505
previous_state1: []
lev_fea: []
======Model training from:  2009-01-01 to  2015-09-28
======A2C Training========
{'n_steps': 5, 'ent_coef': 0.005, 'learning_rate': 0.0007}
Using cpu device
Logging to tensorboard_log/a2c\a2c_88_1
----------------------------------------
| time/                 |              |
|    fps                | 187          |
|    iterations         | 100          |
|    time_elapsed       | 2            |
|    total_timesteps    | 500          |
| train/                |              |
|    entropy_loss       | -42.3        |
|    explained_variance | 0            |
|    learning_rate      | 0.0007       |
|    n_updates          | 99           |
|    policy_loss        | 0.0016       |
|    reward             | 6.681631e-11 |
|    std                | 1.04         |
|    value_loss         | 2.29e-09     |
----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 172            |
|    iterations         | 200            |
|    time_elapsed       | 5              |
|    total_timesteps    | 1000           |
| train/                |                |
|    entropy_loss       | -43.5          |
|    explained_variance | -4.51          |
|    learning_rate      | 0.0007         |
|    n_updates          | 199            |
|    policy_loss        | -26.4          |
|    reward             | -1.8768197e-10 |
|    std                | 1.09           |
|    value_loss         | 0.505          |
------------------------------------------
----------------------------------------
| time/                 |              |
|    fps                | 157          |
|    iterations         | 300          |
|    time_elapsed       | 9            |
|    total_timesteps    | 1500         |
| train/                |              |
|    entropy_loss       | -44.5        |
|    explained_variance | -1.19e-07    |
|    learning_rate      | 0.0007       |
|    n_updates          | 299          |
|    policy_loss        | 0.00901      |
|    reward             | 6.852053e-10 |
|    std                | 1.12         |
|    value_loss         | 7.36e-08     |
----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 159           |
|    iterations         | 400           |
|    time_elapsed       | 12            |
|    total_timesteps    | 2000          |
| train/                |               |
|    entropy_loss       | -45.4         |
|    explained_variance | -1.19e-07     |
|    learning_rate      | 0.0007        |
|    n_updates          | 399           |
|    policy_loss        | -1.83e-06     |
|    reward             | 2.0092133e-11 |
|    std                | 1.16          |
|    value_loss         | 6.48e-16      |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 160           |
|    iterations         | 500           |
|    time_elapsed       | 15            |
|    total_timesteps    | 2500          |
| train/                |               |
|    entropy_loss       | -46.6         |
|    explained_variance | 0             |
|    learning_rate      | 0.0007        |
|    n_updates          | 499           |
|    policy_loss        | -4.26e-08     |
|    reward             | -6.693916e-11 |
|    std                | 1.21          |
|    value_loss         | 1.42e-18      |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 157            |
|    iterations         | 600            |
|    time_elapsed       | 19             |
|    total_timesteps    | 3000           |
| train/                |                |
|    entropy_loss       | -48.2          |
|    explained_variance | 1.79e-07       |
|    learning_rate      | 0.0007         |
|    n_updates          | 599            |
|    policy_loss        | -8.26e-07      |
|    reward             | -5.9255094e-11 |
|    std                | 1.28           |
|    value_loss         | 1.68e-15       |
------------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 155            |
|    iterations         | 700            |
|    time_elapsed       | 22             |
|    total_timesteps    | 3500           |
| train/                |                |
|    entropy_loss       | -49.3          |
|    explained_variance | 0              |
|    learning_rate      | 0.0007         |
|    n_updates          | 699            |
|    policy_loss        | 0.0912         |
|    reward             | -1.5965777e-11 |
|    std                | 1.33           |
|    value_loss         | 5.78e-06       |
------------------------------------------
-------------------------------------------
| time/                 |                 |
|    fps                | 158             |
|    iterations         | 800             |
|    time_elapsed       | 25              |
|    total_timesteps    | 4000            |
| train/                |                 |
|    entropy_loss       | -50.5           |
|    explained_variance | 0               |
|    learning_rate      | 0.0007          |
|    n_updates          | 799             |
|    policy_loss        | 2.31e-07        |
|    reward             | -1.15066554e-10 |
|    std                | 1.38            |
|    value_loss         | 4.23e-15        |
-------------------------------------------
----------------------------------------
| time/                 |              |
|    fps                | 157          |
|    iterations         | 900          |
|    time_elapsed       | 28           |
|    total_timesteps    | 4500         |
| train/                |              |
|    entropy_loss       | -51.9        |
|    explained_variance | -1.76e+03    |
|    learning_rate      | 0.0007       |
|    n_updates          | 899          |
|    policy_loss        | -0.96        |
|    reward             | 1.639947e-10 |
|    std                | 1.45         |
|    value_loss         | 0.00142      |
----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 154           |
|    iterations         | 1000          |
|    time_elapsed       | 32            |
|    total_timesteps    | 5000          |
| train/                |               |
|    entropy_loss       | -52.3         |
|    explained_variance | -3.37e+03     |
|    learning_rate      | 0.0007        |
|    n_updates          | 999           |
|    policy_loss        | 2.45          |
|    reward             | -7.367802e-10 |
|    std                | 1.47          |
|    value_loss         | 0.00852       |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 156            |
|    iterations         | 1100           |
|    time_elapsed       | 35             |
|    total_timesteps    | 5500           |
| train/                |                |
|    entropy_loss       | -52.6          |
|    explained_variance | 0              |
|    learning_rate      | 0.0007         |
|    n_updates          | 1099           |
|    policy_loss        | -3.71          |
|    reward             | -1.7347441e-09 |
|    std                | 1.49           |
|    value_loss         | 0.00652        |
------------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 157            |
|    iterations         | 1200           |
|    time_elapsed       | 38             |
|    total_timesteps    | 6000           |
| train/                |                |
|    entropy_loss       | -53            |
|    explained_variance | -2.7           |
|    learning_rate      | 0.0007         |
|    n_updates          | 1199           |
|    policy_loss        | 0.0568         |
|    reward             | -1.8465131e-10 |
|    std                | 1.5            |
|    value_loss         | 7.79e-06       |
------------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 155           |
|    iterations         | 1300          |
|    time_elapsed       | 41            |
|    total_timesteps    | 6500          |
| train/                |               |
|    entropy_loss       | -53.3         |
|    explained_variance | -166          |
|    learning_rate      | 0.0007        |
|    n_updates          | 1299          |
|    policy_loss        | 2.4           |
|    reward             | 1.3847642e-09 |
|    std                | 1.52          |
|    value_loss         | 0.00597       |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 155           |
|    iterations         | 1400          |
|    time_elapsed       | 44            |
|    total_timesteps    | 7000          |
| train/                |               |
|    entropy_loss       | -53.6         |
|    explained_variance | -5.11e+03     |
|    learning_rate      | 0.0007        |
|    n_updates          | 1399          |
|    policy_loss        | -4.3          |
|    reward             | 2.3425135e-09 |
|    std                | 1.54          |
|    value_loss         | 0.0133        |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 157           |
|    iterations         | 1500          |
|    time_elapsed       | 47            |
|    total_timesteps    | 7500          |
| train/                |               |
|    entropy_loss       | -54           |
|    explained_variance | 0             |
|    learning_rate      | 0.0007        |
|    n_updates          | 1499          |
|    policy_loss        | -0.0794       |
|    reward             | -7.215917e-09 |
|    std                | 1.56          |
|    value_loss         | 3.51e-06      |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 156            |
|    iterations         | 1600           |
|    time_elapsed       | 51             |
|    total_timesteps    | 8000           |
| train/                |                |
|    entropy_loss       | -54.6          |
|    explained_variance | -57.2          |
|    learning_rate      | 0.0007         |
|    n_updates          | 1599           |
|    policy_loss        | -1.35          |
|    reward             | -2.0928562e-10 |
|    std                | 1.59           |
|    value_loss         | 0.00137        |
------------------------------------------
day: 1694, episode: 5
begin_total_asset: 1000000.00
end_total_asset: 3332677.16
total_reward: 2332677.16
total_cost: 72569.96
total_trades: 35068
Sharpe: 1.044
=================================
--------------------------------------
| time/                 |            |
|    fps                | 155        |
|    iterations         | 1700       |
|    time_elapsed       | 54         |
|    total_timesteps    | 8500       |
| train/                |            |
|    entropy_loss       | -55.2      |
|    explained_variance | 0          |
|    learning_rate      | 0.0007     |
|    n_updates          | 1699       |
|    policy_loss        | -22.3      |
|    reward             | 0.09996601 |
|    std                | 1.63       |
|    value_loss         | 0.242      |
--------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 156           |
|    iterations         | 1800          |
|    time_elapsed       | 57            |
|    total_timesteps    | 9000          |
| train/                |               |
|    entropy_loss       | -55.5         |
|    explained_variance | -29.7         |
|    learning_rate      | 0.0007        |
|    n_updates          | 1799          |
|    policy_loss        | 0.225         |
|    reward             | 4.0747247e-10 |
|    std                | 1.64          |
|    value_loss         | 0.000747      |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 156            |
|    iterations         | 1900           |
|    time_elapsed       | 60             |
|    total_timesteps    | 9500           |
| train/                |                |
|    entropy_loss       | -55.8          |
|    explained_variance | -108           |
|    learning_rate      | 0.0007         |
|    n_updates          | 1899           |
|    policy_loss        | -0.31          |
|    reward             | -7.9517705e-11 |
|    std                | 1.66           |
|    value_loss         | 0.00048        |
------------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 155            |
|    iterations         | 2000           |
|    time_elapsed       | 64             |
|    total_timesteps    | 10000          |
| train/                |                |
|    entropy_loss       | -56.3          |
|    explained_variance | -1.38          |
|    learning_rate      | 0.0007         |
|    n_updates          | 1999           |
|    policy_loss        | -2.16          |
|    reward             | -6.1809974e-10 |
|    std                | 1.69           |
|    value_loss         | 0.00194        |
------------------------------------------
======A2C Validation from:  2015-09-28 to  2015-11-30
previous_state1: []
lev_fea: []
start borrow! 0 128.26981985772406 day 0
sell 12
start borrow! 59.33313851462354 128.26981985772406 day 13
sell 18
start borrow! 42.19158951709052 128.26981985772406 day 20
A2C Sharpe Ratio:  0.16287314688433863
======PPO Training========
{'ent_coef': 0.01, 'n_steps': 2048, 'learning_rate': 0.00025, 'batch_size': 128}
Using cpu device
Logging to tensorboard_log/ppo\ppo_88_1
--------------------------------------
| time/              |               |
|    fps             | 169           |
|    iterations      | 1             |
|    time_elapsed    | 12            |
|    total_timesteps | 2048          |
| train/             |               |
|    reward          | -5.368614e-11 |
--------------------------------------
-------------------------------------------
| time/                   |               |
|    fps                  | 164           |
|    iterations           | 2             |
|    time_elapsed         | 24            |
|    total_timesteps      | 4096          |
| train/                  |               |
|    approx_kl            | 0.04681784    |
|    clip_fraction        | 0.501         |
|    clip_range           | 0.2           |
|    entropy_loss         | -41.3         |
|    explained_variance   | -0.0202       |
|    learning_rate        | 0.00025       |
|    loss                 | -0.289        |
|    n_updates            | 10            |
|    policy_gradient_loss | -0.0316       |
|    reward               | 1.3104323e-10 |
|    std                  | 1.01          |
|    value_loss           | 0.359         |
-------------------------------------------
-------------------------------------------
| time/                   |               |
|    fps                  | 163           |
|    iterations           | 3             |
|    time_elapsed         | 37            |
|    total_timesteps      | 6144          |
| train/                  |               |
|    approx_kl            | 0.047698792   |
|    clip_fraction        | 0.434         |
|    clip_range           | 0.2           |
|    entropy_loss         | -41.4         |
|    explained_variance   | -0.0115       |
|    learning_rate        | 0.00025       |
|    loss                 | -0.279        |
|    n_updates            | 20            |
|    policy_gradient_loss | -0.0157       |
|    reward               | 1.7754219e-10 |
|    std                  | 1.01          |
|    value_loss           | 0.166         |
-------------------------------------------
day: 1694, episode: 10
begin_total_asset: 1000000.00
end_total_asset: 1616079.68
total_reward: 616079.68
total_cost: 104539.22
total_trades: 42068
Sharpe: 0.722
=================================
--------------------------------------------
| time/                   |                |
|    fps                  | 161            |
|    iterations           | 4              |
|    time_elapsed         | 50             |
|    total_timesteps      | 8192           |
| train/                  |                |
|    approx_kl            | 0.04880053     |
|    clip_fraction        | 0.49           |
|    clip_range           | 0.2            |
|    entropy_loss         | -41.3          |
|    explained_variance   | -0.0096        |
|    learning_rate        | 0.00025        |
|    loss                 | -0.324         |
|    n_updates            | 30             |
|    policy_gradient_loss | -0.0159        |
|    reward               | 1.08650956e-10 |
|    std                  | 1              |
|    value_loss           | 0.193          |
--------------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 160        |
|    iterations           | 5          |
|    time_elapsed         | 63         |
|    total_timesteps      | 10240      |
| train/                  |            |
|    approx_kl            | 0.03991487 |
|    clip_fraction        | 0.52       |
|    clip_range           | 0.2        |
|    entropy_loss         | -41.3      |
|    explained_variance   | -0.00834   |
|    learning_rate        | 0.00025    |
|    loss                 | -0.183     |
|    n_updates            | 40         |
|    policy_gradient_loss | -0.0214    |
|    reward               | 1.5382168  |
|    std                  | 1          |
|    value_loss           | 0.236      |
----------------------------------------
======PPO Validation from:  2015-09-28 to  2015-11-30
previous_state1: []
lev_fea: []
start borrow! 0 128.26981985772406 day 0
sell 12
start borrow! 59.33313851462354 128.26981985772406 day 13
sell 18
start borrow! 42.19158951709052 128.26981985772406 day 20
PPO Sharpe Ratio:  0.10183769928603316
======DDPG Training========
{'buffer_size': 10000, 'learning_rate': 0.0005, 'batch_size': 64}
Using cpu device
Logging to tensorboard_log/ddpg\ddpg_88_1
day: 1694, episode: 15
begin_total_asset: 1000000.00
end_total_asset: 2117030.47
total_reward: 1117030.47
total_cost: 1132.22
total_trades: 25708
Sharpe: 0.748
=================================
--------------------------------------
| time/              |               |
|    episodes        | 4             |
|    fps             | 69            |
|    time_elapsed    | 97            |
|    total_timesteps | 6780          |
| train/             |               |
|    actor_loss      | 2.73          |
|    critic_loss     | 2.04e+03      |
|    learning_rate   | 0.0005        |
|    n_updates       | 5085          |
|    reward          | 8.4345037e-10 |
--------------------------------------
======DDPG Validation from:  2015-09-28 to  2015-11-30
previous_state1: []
lev_fea: []
start borrow! 0 128.26981985772406 day 0
sell 12
start borrow! 59.33313851462354 128.26981985772406 day 13
sell 18
start borrow! 42.19158951709052 128.26981985772406 day 20
======Best Model Retraining from:  2009-01-01 to  2015-11-30
======Trading from:  2015-11-30 to  2016-02-03
previous_state1: []
lev_fea: []
start borrow! 0 128.26981985772406 day 0
sell 11
start borrow! 70.96367759568794 128.26981985772406 day 12
sell 36
start borrow! 31.277224405153095 128.26981985772406 day 37
============================================
turbulence_threshold:  160.33727482215505
previous_state1: []
lev_fea: tensor([6.0000e+00, 1.0000e+00, 9.3507e+05], dtype=torch.float64)
======Model training from:  2009-01-01 to  2015-11-30
======A2C Training========
{'n_steps': 5, 'ent_coef': 0.005, 'learning_rate': 0.0007}
Using cpu device
Logging to tensorboard_log/a2c\a2c_132_1
-----------------------------------------
| time/                 |               |
|    fps                | 159           |
|    iterations         | 100           |
|    time_elapsed       | 3             |
|    total_timesteps    | 500           |
| train/                |               |
|    entropy_loss       | -42.2         |
|    explained_variance | -4.05e+04     |
|    learning_rate      | 0.0007        |
|    n_updates          | 99            |
|    policy_loss        | 5.74          |
|    reward             | -2.537883e-10 |
|    std                | 1.04          |
|    value_loss         | 0.0365        |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 147            |
|    iterations         | 200            |
|    time_elapsed       | 6              |
|    total_timesteps    | 1000           |
| train/                |                |
|    entropy_loss       | -42.7          |
|    explained_variance | -1.04          |
|    learning_rate      | 0.0007         |
|    n_updates          | 199            |
|    policy_loss        | 0.278          |
|    reward             | -1.1950192e-09 |
|    std                | 1.05           |
|    value_loss         | 8.15e-05       |
------------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 134           |
|    iterations         | 300           |
|    time_elapsed       | 11            |
|    total_timesteps    | 1500          |
| train/                |               |
|    entropy_loss       | -43.2         |
|    explained_variance | -6.87e+07     |
|    learning_rate      | 0.0007        |
|    n_updates          | 299           |
|    policy_loss        | -6.49         |
|    reward             | 1.5433336e-09 |
|    std                | 1.07          |
|    value_loss         | 0.0778        |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 133           |
|    iterations         | 400           |
|    time_elapsed       | 14            |
|    total_timesteps    | 2000          |
| train/                |               |
|    entropy_loss       | -43.7         |
|    explained_variance | -649          |
|    learning_rate      | 0.0007        |
|    n_updates          | 399           |
|    policy_loss        | 1.66          |
|    reward             | -4.573734e-10 |
|    std                | 1.09          |
|    value_loss         | 0.00625       |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 132           |
|    iterations         | 500           |
|    time_elapsed       | 18            |
|    total_timesteps    | 2500          |
| train/                |               |
|    entropy_loss       | -44.3         |
|    explained_variance | 0             |
|    learning_rate      | 0.0007        |
|    n_updates          | 499           |
|    policy_loss        | -0.24         |
|    reward             | 1.8687443e-09 |
|    std                | 1.11          |
|    value_loss         | 5.44e-05      |
-----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 127           |
|    iterations         | 600           |
|    time_elapsed       | 23            |
|    total_timesteps    | 3000          |
| train/                |               |
|    entropy_loss       | -45           |
|    explained_variance | -98.3         |
|    learning_rate      | 0.0007        |
|    n_updates          | 599           |
|    policy_loss        | -1.29         |
|    reward             | 3.0666325e-10 |
|    std                | 1.14          |
|    value_loss         | 0.00246       |
-----------------------------------------
---------------------------------------
| time/                 |             |
|    fps                | 123         |
|    iterations         | 700         |
|    time_elapsed       | 28          |
|    total_timesteps    | 3500        |
| train/                |             |
|    entropy_loss       | -45.8       |
|    explained_variance | 0           |
|    learning_rate      | 0.0007      |
|    n_updates          | 699         |
|    policy_loss        | 8.19        |
|    reward             | -0.18678509 |
|    std                | 1.17        |
|    value_loss         | 0.044       |
---------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 126            |
|    iterations         | 800            |
|    time_elapsed       | 31             |
|    total_timesteps    | 4000           |
| train/                |                |
|    entropy_loss       | -46.2          |
|    explained_variance | 0              |
|    learning_rate      | 0.0007         |
|    n_updates          | 799            |
|    policy_loss        | -0.0624        |
|    reward             | -1.0688088e-09 |
|    std                | 1.19           |
|    value_loss         | 3.24e-06       |
------------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 127           |
|    iterations         | 900           |
|    time_elapsed       | 35            |
|    total_timesteps    | 4500          |
| train/                |               |
|    entropy_loss       | -46.9         |
|    explained_variance | -67.8         |
|    learning_rate      | 0.0007        |
|    n_updates          | 899           |
|    policy_loss        | -0.143        |
|    reward             | 1.1789741e-09 |
|    std                | 1.22          |
|    value_loss         | 0.00192       |
-----------------------------------------
----------------------------------------
| time/                 |              |
|    fps                | 125          |
|    iterations         | 1000         |
|    time_elapsed       | 39           |
|    total_timesteps    | 5000         |
| train/                |              |
|    entropy_loss       | -47.4        |
|    explained_variance | -513         |
|    learning_rate      | 0.0007       |
|    n_updates          | 999          |
|    policy_loss        | -4.43        |
|    reward             | 9.393726e-10 |
|    std                | 1.24         |
|    value_loss         | 0.0305       |
----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 125           |
|    iterations         | 1100          |
|    time_elapsed       | 43            |
|    total_timesteps    | 5500          |
| train/                |               |
|    entropy_loss       | -47.7         |
|    explained_variance | -1.77e+05     |
|    learning_rate      | 0.0007        |
|    n_updates          | 1099          |
|    policy_loss        | -5.83         |
|    reward             | 1.1277875e-09 |
|    std                | 1.25          |
|    value_loss         | 0.0534        |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 126            |
|    iterations         | 1200           |
|    time_elapsed       | 47             |
|    total_timesteps    | 6000           |
| train/                |                |
|    entropy_loss       | -48.3          |
|    explained_variance | -96.7          |
|    learning_rate      | 0.0007         |
|    n_updates          | 1199           |
|    policy_loss        | -0.12          |
|    reward             | -6.6301437e-10 |
|    std                | 1.28           |
|    value_loss         | 3.39e-05       |
------------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 125           |
|    iterations         | 1300          |
|    time_elapsed       | 51            |
|    total_timesteps    | 6500          |
| train/                |               |
|    entropy_loss       | -48.9         |
|    explained_variance | -1.24e+03     |
|    learning_rate      | 0.0007        |
|    n_updates          | 1299          |
|    policy_loss        | -2.06         |
|    reward             | 1.0956612e-09 |
|    std                | 1.31          |
|    value_loss         | 0.0135        |
-----------------------------------------
--------------------------------------
| time/                 |            |
|    fps                | 124        |
|    iterations         | 1400       |
|    time_elapsed       | 56         |
|    total_timesteps    | 7000       |
| train/                |            |
|    entropy_loss       | -49.5      |
|    explained_variance | 0          |
|    learning_rate      | 0.0007     |
|    n_updates          | 1399       |
|    policy_loss        | -42.4      |
|    reward             | 0.06003343 |
|    std                | 1.33       |
|    value_loss         | 1.13       |
--------------------------------------
----------------------------------------
| time/                 |              |
|    fps                | 125          |
|    iterations         | 1500         |
|    time_elapsed       | 59           |
|    total_timesteps    | 7500         |
| train/                |              |
|    entropy_loss       | -49.8        |
|    explained_variance | -2.28        |
|    learning_rate      | 0.0007       |
|    n_updates          | 1499         |
|    policy_loss        | -0.201       |
|    reward             | 9.052265e-10 |
|    std                | 1.35         |
|    value_loss         | 0.000104     |
----------------------------------------
-----------------------------------------
| time/                 |               |
|    fps                | 124           |
|    iterations         | 1600          |
|    time_elapsed       | 64            |
|    total_timesteps    | 8000          |
| train/                |               |
|    entropy_loss       | -50.3         |
|    explained_variance | -1.22e+06     |
|    learning_rate      | 0.0007        |
|    n_updates          | 1599          |
|    policy_loss        | 0.443         |
|    reward             | 3.5695742e-09 |
|    std                | 1.37          |
|    value_loss         | 0.00223       |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 122            |
|    iterations         | 1700           |
|    time_elapsed       | 69             |
|    total_timesteps    | 8500           |
| train/                |                |
|    entropy_loss       | -50.9          |
|    explained_variance | 0              |
|    learning_rate      | 0.0007         |
|    n_updates          | 1699           |
|    policy_loss        | 0.213          |
|    reward             | -1.3853484e-09 |
|    std                | 1.4            |
|    value_loss         | 3.17e-05       |
------------------------------------------
day: 1738, episode: 5
begin_total_asset: 1000000.00
end_total_asset: 3402521.95
total_reward: 2402521.95
total_cost: 76475.09
total_trades: 35858
Sharpe: 0.968
=================================
-----------------------------------------
| time/                 |               |
|    fps                | 122           |
|    iterations         | 1800          |
|    time_elapsed       | 73            |
|    total_timesteps    | 9000          |
| train/                |               |
|    entropy_loss       | -51.2         |
|    explained_variance | -194          |
|    learning_rate      | 0.0007        |
|    n_updates          | 1799          |
|    policy_loss        | -0.0638       |
|    reward             | 4.0625442e-10 |
|    std                | 1.42          |
|    value_loss         | 0.000107      |
-----------------------------------------
------------------------------------------
| time/                 |                |
|    fps                | 122            |
|    iterations         | 1900           |
|    time_elapsed       | 77             |
|    total_timesteps    | 9500           |
| train/                |                |
|    entropy_loss       | -51.8          |
|    explained_variance | -405           |
|    learning_rate      | 0.0007         |
|    n_updates          | 1899           |
|    policy_loss        | -0.962         |
|    reward             | -2.0853377e-10 |
|    std                | 1.45           |
|    value_loss         | 0.00123        |
------------------------------------------
----------------------------------------
| time/                 |              |
|    fps                | 121          |
|    iterations         | 2000         |
|    time_elapsed       | 82           |
|    total_timesteps    | 10000        |
| train/                |              |
|    entropy_loss       | -52.2        |
|    explained_variance | -34.1        |
|    learning_rate      | 0.0007       |
|    n_updates          | 1999         |
|    policy_loss        | 7.76         |
|    reward             | 4.711339e-10 |
|    std                | 1.46         |
|    value_loss         | 0.0319       |
----------------------------------------
======A2C Validation from:  2015-11-30 to  2016-02-03
previous_state1: []
lev_fea: tensor([6.0000e+00, 1.0000e+00, 9.3507e+05], dtype=torch.float64)
start borrow! 0 128.26981985772406 day 0
sell 11
start borrow! 70.96367759568794 128.26981985772406 day 12
sell 36
start borrow! 31.277224405153095 128.26981985772406 day 37
A2C Sharpe Ratio:  -0.35546505547529644
======PPO Training========
{'ent_coef': 0.01, 'n_steps': 2048, 'learning_rate': 0.00025, 'batch_size': 128}
Using cpu device
Logging to tensorboard_log/ppo\ppo_132_1
---------------------------------------
| time/              |                |
|    fps             | 97             |
|    iterations      | 1              |
|    time_elapsed    | 21             |
|    total_timesteps | 2048           |
| train/             |                |
|    reward          | -6.6502276e-11 |
---------------------------------------
-------------------------------------------
| time/                   |               |
|    fps                  | 106           |
|    iterations           | 2             |
|    time_elapsed         | 38            |
|    total_timesteps      | 4096          |
| train/                  |               |
|    approx_kl            | 0.038372617   |
|    clip_fraction        | 0.484         |
|    clip_range           | 0.2           |
|    entropy_loss         | -41.3         |
|    explained_variance   | -0.125        |
|    learning_rate        | 0.00025       |
|    loss                 | -0.219        |
|    n_updates            | 10            |
|    policy_gradient_loss | -0.045        |
|    reward               | 1.1501554e-10 |
|    std                  | 1.01          |
|    value_loss           | 0.507         |
-------------------------------------------
-------------------------------------------
| time/                   |               |
|    fps                  | 111           |
|    iterations           | 3             |
|    time_elapsed         | 54            |
|    total_timesteps      | 6144          |
| train/                  |               |
|    approx_kl            | 0.032746956   |
|    clip_fraction        | 0.451         |
|    clip_range           | 0.2           |
|    entropy_loss         | -41.3         |
|    explained_variance   | -0.0115       |
|    learning_rate        | 0.00025       |
|    loss                 | -0.353        |
|    n_updates            | 20            |
|    policy_gradient_loss | -0.0262       |
|    reward               | 7.1864986e-10 |
|    std                  | 1.01          |
|    value_loss           | 0.268         |
-------------------------------------------
day: 1738, episode: 10
begin_total_asset: 1000000.00
end_total_asset: 1688045.67
total_reward: 688045.67
total_cost: 123049.68
total_trades: 40517
Sharpe: 0.798
=================================
------------------------------------------
| time/                   |              |
|    fps                  | 114          |
|    iterations           | 4            |
|    time_elapsed         | 71           |
|    total_timesteps      | 8192         |
| train/                  |              |
|    approx_kl            | 0.034090653  |
|    clip_fraction        | 0.486        |
|    clip_range           | 0.2          |
|    entropy_loss         | -41.3        |
|    explained_variance   | -0.0039      |
|    learning_rate        | 0.00025      |
|    loss                 | -0.305       |
|    n_updates            | 30           |
|    policy_gradient_loss | -0.0258      |
|    reward               | 6.785686e-11 |
|    std                  | 1            |
|    value_loss           | 0.297        |
------------------------------------------
-------------------------------------------
| time/                   |               |
|    fps                  | 115           |
|    iterations           | 5             |
|    time_elapsed         | 88            |
|    total_timesteps      | 10240         |
| train/                  |               |
|    approx_kl            | 0.029621862   |
|    clip_fraction        | 0.489         |
|    clip_range           | 0.2           |
|    entropy_loss         | -41.2         |
|    explained_variance   | -0.00747      |
|    learning_rate        | 0.00025       |
|    loss                 | -0.329        |
|    n_updates            | 40            |
|    policy_gradient_loss | -0.0193       |
|    reward               | 2.2062024e-10 |
|    std                  | 0.998         |
|    value_loss           | 0.277         |
-------------------------------------------
======PPO Validation from:  2015-11-30 to  2016-02-03
previous_state1: []
lev_fea: tensor([6.0000e+00, 1.0000e+00, 9.3507e+05], dtype=torch.float64)
start borrow! 0 128.26981985772406 day 0
sell 11
start borrow! 70.96367759568794 128.26981985772406 day 12
sell 36
start borrow! 31.277224405153095 128.26981985772406 day 37
PPO Sharpe Ratio:  -0.5539601821916942
======DDPG Training========
{'buffer_size': 10000, 'learning_rate': 0.0005, 'batch_size': 64}
Using cpu device
Logging to tensorboard_log/ddpg\ddpg_132_1
