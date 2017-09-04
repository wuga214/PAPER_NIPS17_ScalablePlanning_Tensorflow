Fast Running Code for AAAI-18
===

# Version
Tensorflow 1.3

# How to reproduce paper results
python main.py
python reservoir_timing.py

# Results
```
NAVOptimizer NAVI_BILINEAR NAVI_05_STRIDE Planning Step:30
100%|███████████████████████████████████████████████| 1000/1000 [00:37<00:00, 15.37it/s]
mean: -185.332839966, std: 0.139176920056
NAVOptimizer NAVI_BILINEAR NAVI_03_STRIDE Planning Step:60
100%|███████████████████████████████████████████████| 1000/1000 [01:59<00:00,  7.25it/s]
mean: -310.039489746, std: 0.253329247236
NAVOptimizer NAVI_BILINEAR NAVI_02_STRIDE Planning Step:120
100%|███████████████████████████████████████████████| 1000/1000 [03:39<00:00,  4.43it/s]
mean: -492.074645996, std: 0.576007843018
NAVOptimizer NAVI_NONLINEAR NAVI_05_STRIDE Planning Step:30
100%|███████████████████████████████████████████████| 1000/1000 [04:58<00:00,  3.36it/s]
mean: -153.452911377, std: 0.131727769971
NAVOptimizer NAVI_NONLINEAR NAVI_03_STRIDE Planning Step:60
100%|███████████████████████████████████████████████| 1000/1000 [06:52<00:00,  2.49it/s]
mean: -254.959716797, std: 0.799005687237
NAVOptimizer NAVI_NONLINEAR NAVI_02_STRIDE Planning Step:120
100%|███████████████████████████████████████████████| 1000/1000 [08:53<00:00,  1.92it/s]
mean: -399.532501221, std: 1.61619126797
NAVOptimizer NAVI_LINEAR NAVI_05_STRIDE Planning Step:30
100%|███████████████████████████████████████████████| 1000/1000 [41:59<00:00,  2.43s/it]
mean: -176.704605103, std: 4.10427618027
NAVOptimizer NAVI_LINEAR NAVI_03_STRIDE Planning Step:60
100%|█████████████████████████████████████████████| 1000/1000 [1:23:14<00:00,  4.97s/it]
mean: -437.396789551, std: 79.4487838745
NAVOptimizer NAVI_LINEAR NAVI_02_STRIDE Planning Step:120
100%|█████████████████████████████████████████████| 1000/1000 [2:09:44<00:00,  7.55s/it]
mean: -1004.60040283, std: 6.103515625e-05
ReservoirOptimizer RESERVOIR_LINEAR RESE_20_RESERVOIRS Planning Step:30
100%|█████████████████████████████████████████████████| 500/500 [42:49<00:00,  5.25s/it]
mean: -1802.6340332, std: 0.899782896042
ReservoirOptimizer RESERVOIR_LINEAR RESE_20_RESERVOIRS Planning Step:60
100%|█████████████████████████████████████████████████| 500/500 [44:13<00:00,  5.35s/it]
mean: -2280.12768555, std: 0.769890487194
ReservoirOptimizer RESERVOIR_LINEAR RESE_20_RESERVOIRS Planning Step:120
100%|█████████████████████████████████████████████████| 500/500 [45:22<00:00,  5.41s/it]
mean: -3123.19555664, std: 1.03462922573
ReservoirOptimizer RESERVOIR_NONLINEAR RESE_20_RESERVOIRS Planning Step:30
100%|█████████████████████████████████████████████████| 500/500 [45:16<00:00,  5.46s/it]
mean: -1882.33166504, std: 1.53480947018
ReservoirOptimizer RESERVOIR_NONLINEAR RESE_20_RESERVOIRS Planning Step:60
100%|█████████████████████████████████████████████████| 500/500 [46:10<00:00,  5.49s/it]
mean: -3601.55859375, std: 2.2363319397
ReservoirOptimizer RESERVOIR_NONLINEAR RESE_20_RESERVOIRS Planning Step:120
100%|█████████████████████████████████████████████████| 500/500 [47:06<00:00,  5.68s/it]
mean: -7088.21191406, std: 3.82561349869
HVACOptimizer HVAC HVAC_60_ROOMS Planning Step:12
100%|███████████████████████████████████████████████| 2000/2000 [02:52<00:00,  6.73it/s]
mean: -18678.7460938, std: 3.93702697754
HVACOptimizer HVAC HVAC_60_ROOMS Planning Step:24
100%|███████████████████████████████████████████████| 2000/2000 [04:01<00:00,  5.40it/s]
mean: -20434.3867188, std: 13.9654216766
HVACOptimizer HVAC HVAC_60_ROOMS Planning Step:48
100%|███████████████████████████████████████████████| 2000/2000 [05:26<00:00,  3.67it/s]
mean: -25491.59375, std: 34.5582733154
HVACOptimizer HVAC HVAC_60_ROOMS Planning Step:96
100%|███████████████████████████████████████████████| 2000/2000 [08:08<00:00,  3.40it/s]
mean: -39089.8828125, std: 560.183776855
```
