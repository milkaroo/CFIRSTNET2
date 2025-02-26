# CFIRSTNET: Comprehensive Features for Static IR Drop Estimation with Neural Network

Official implementation of **[CFIRSTNET: Comprehensive Features for Static IR Drop Estimation with Neural Network](<https://arxiv.org/abs/2502.12168>)**.
<p align='center'>
<td style='text-align:center;'>
  <img src=https://github.com/jason122490/CFIRSTNET/blob/main/figures/flow.png >
</td>
</p>

## Introduction

We propose a comprehensive solution to combine both image-based and netlist-based features in neural network framework and obtain high-quality static IR drop prediction very effectively in modern designs. A customized convolutional neural network (CNN) is developed to extract PDN features and make static IR drop estimations.
<p align='center'>
<td style='text-align:center;'>
  <img src=https://github.com/jason122490/CFIRSTNET/blob/main/figures/model.png >
</td>
</p>
Trained and evaluated with the open-source dataset, experiment results show that we have obtained the best quality in the benchmark on the problem of IR drop estimation in ICCAD CAD Contest 2023.

<table><thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="4" rowspan="2">HSPICE</th>
    <th colspan="4" rowspan="2">IREDGe</th>
    <th colspan="4" rowspan="2">1st Place of CAD Contest</th>
    <th colspan="5" rowspan="2">CFIRSTNET</th>
  </tr>
  <tr>
  </tr></thead>
<tbody>
  <tr>
    <td>Instance</td>
    <td>$IR_{avg}$<br>(mV)</td>
    <td>$IR_{max}$<br>(mV)</td>
    <td>Hotspot<br>(%)</td>
    <td>Runtime<br>(sec)</td>
    <td>$e_{avg}$<br>(mV)</td>
    <td>$e_{max}$<br>(mV)</td>
    <td>F1 Score</td>
    <td>runtime<br>(sec)</td>
    <td>$e_{avg}$<br>(mV)</td>
    <td>$e_{max}$<br>(mV)</td>
    <td>F1 Score</td>
    <td>Runtime<br>(sec)</td>
    <td>$e_{avg}$<br>(mV)</td>
    <td>$e_{max}$<br>(mV)</td>
    <td>F1 Score</td>
    <td>Runtime<br>(sec)</td>
    <td>Speedup<br>(vs HSPICE)</td>
  </tr>
  <tr>
    <td>testcase7</td>
    <td>1.0439</td>
    <td>4.3045</td>
    <td>0.378</td>
    <td>8.94</td>
    <td>0.6218</td>
    <td>1.2305</td>
    <td>0.142</td>
    <td>0.150</td>
    <td>0.0656</td>
    <td>1.2115</td>
    <td>0.783</td>
    <td>7.996</td>
    <td>0.0177</td>
    <td>0.3458</td>
    <td>0.923</td>
    <td>0.366</td>
    <td>24.43x</td>
  </tr>
  <tr>
    <td>testcase8</td>
    <td>1.5540</td>
    <td>4.8994</td>
    <td>0.535</td>
    <td>8.86</td>
    <td>0.3845</td>
    <td>2.2659</td>
    <td>0.419</td>
    <td>0.149</td>
    <td>0.0815</td>
    <td>1.0416</td>
    <td>0.816</td>
    <td>8.396</td>
    <td>0.0257</td>
    <td>0.5214</td>
    <td>0.916</td>
    <td>0.367</td>
    <td>24.14x</td>
  </tr>
  <tr>
    <td>testcase9</td>
    <td>1.1811</td>
    <td>3.7932</td>
    <td>0.034</td>
    <td>21.07</td>
    <td>0.4538</td>
    <td>1.2780</td>
    <td>0</td>
    <td>0.264</td>
    <td>0.0406</td>
    <td>0.8755</td>
    <td>0.589</td>
    <td>11.417</td>
    <td>0.0278</td>
    <td>0.4664</td>
    <td>0.526</td>
    <td>0.572</td>
    <td>36.84x</td>
  </tr>
  <tr>
    <td>testcase10</td>
    <td>1.9483</td>
    <td>4.5327</td>
    <td>0.086</td>
    <td>17.53</td>
    <td>0.2426</td>
    <td>1.2721</td>
    <td>0</td>
    <td>0.262</td>
    <td>0.0659</td>
    <td>0.8547</td>
    <td>0.532</td>
    <td>11.270</td>
    <td>0.0459</td>
    <td>0.5741</td>
    <td>0.464</td>
    <td>0.551</td>
    <td>31.81x</td>
  </tr>
  <tr>
    <td>testcase13</td>
    <td>2.2638</td>
    <td>10.5650</td>
    <td>0.080</td>
    <td>3.22</td>
    <td>0.2441</td>
    <td>3.8389</td>
    <td>0</td>
    <td>0.033</td>
    <td>0.2068</td>
    <td>7.2341</td>
    <td>0</td>
    <td>5.452</td>
    <td>0.0774</td>
    <td>2.1299</td>
    <td>0.680</td>
    <td>0.204</td>
    <td>15.78x</td>
  </tr>
  <tr>
    <td>testcase14</td>
    <td>3.2298</td>
    <td>13.1495</td>
    <td>0.088</td>
    <td>2.81</td>
    <td>0.3138</td>
    <td>5.5321</td>
    <td>0</td>
    <td>0.033</td>
    <td>0.4215</td>
    <td>8.8334</td>
    <td>0</td>
    <td>5.463</td>
    <td>0.1895</td>
    <td>2.1035</td>
    <td>0.678</td>
    <td>0.203</td>
    <td>13.84x</td>
  </tr>
  <tr>
    <td>testcase15</td>
    <td>2.7686</td>
    <td>5.7812</td>
    <td>0.067</td>
    <td>8.86</td>
    <td>0.1530</td>
    <td>1.7691</td>
    <td>0</td>
    <td>0.105</td>
    <td>0.0968</td>
    <td>1.5265</td>
    <td>0.088</td>
    <td>8.137</td>
    <td>0.0353</td>
    <td>0.6735</td>
    <td>0.733</td>
    <td>0.327</td>
    <td>27.09x</td>
  </tr>
  <tr>
    <td>testcase16</td>
    <td>4.6521</td>
    <td>7.5669</td>
    <td>0.345</td>
    <td>8.47</td>
    <td>0.2675</td>
    <td>1.6696</td>
    <td>0.258</td>
    <td>0.102</td>
    <td>0.1601</td>
    <td>1.5487</td>
    <td>0.529</td>
    <td>7.413</td>
    <td>0.0763</td>
    <td>0.8393</td>
    <td>0.785</td>
    <td>0.324</td>
    <td>26.14x</td>
  </tr>
  <tr>
    <td>testcase19</td>
    <td>0.4442</td>
    <td>1.7226</td>
    <td>0.057</td>
    <td>23.33</td>
    <td>1.0649</td>
    <td>1.4689</td>
    <td>0</td>
    <td>0.281</td>
    <td>0.0905</td>
    <td>0.5148</td>
    <td>0.501</td>
    <td>11.905</td>
    <td>0.0187</td>
    <td>0.3187</td>
    <td>0.752</td>
    <td>0.596</td>
    <td>39.14x</td>
  </tr>
  <tr>
    <td>testcase20</td>
    <td>0.6994</td>
    <td>2.4261</td>
    <td>0.010</td>
    <td>18.91</td>
    <td>0.8204</td>
    <td>1.4209</td>
    <td>0</td>
    <td>0.279</td>
    <td>0.1180</td>
    <td>0.5003</td>
    <td>0.711</td>
    <td>11.758</td>
    <td>0.0191</td>
    <td>0.3026</td>
    <td>0.773</td>
    <td>0.576</td>
    <td>32.83x</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>1.9785</td>
    <td>5.8741</td>
    <td>0.168</td>
    <td>12.2</td>
    <td>0.4566</td>
    <td>2.1746</td>
    <td>0.082</td>
    <td>0.166</td>
    <td>0.1347</td>
    <td>2.4141</td>
    <td>0.455</td>
    <td>8.921</td>
    <td>0.0533</td>
    <td>0.8275</td>
    <td>0.723</td>
    <td>0.409</td>
    <td>27.20x</td>
  </tr>
</tbody></table>

## Install & Build

Our codes are seperated into two parts, data preprocess part coding with C++, and the other part including model coding with Python. To run these codes, you must follow the following instructions.

1. Install Conda (Anaconda is tested) and CMake.

2. Create a conda environment with Python 3.10 and activate it.

    ```Command-line
    conda create -n myenv python=3.10 -y
    conda activate myenv
    ```

3. Install requirements for Conda environment.

    ```Command-line
    conda install --file conda_requirements.txt -y
    ```

4. Build the C++ codes for data preprocess with CMake.

    ```Command-line
    cmake -S ./c++ -B ./c++/build
    cmake --build ./c++/build
    cp ./c++/build/libdata.so ./src/libdata.so
    ```

5. Install requirement for Python environment.

    ```Command-line
    pip install -r py_requirements.txt
    ```

6. Run Training

    ```Command-line
    python train.py
    ```
