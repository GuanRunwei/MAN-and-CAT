# MAN and CAT: Mix Attention to NN and Concatenate Attention to YOLO
<div align="center">
  <p>===This is the official page of Mix Attention, MANet and CAT-YOLO===</p>
  <img src="https://github.com/GuanRunwei/MAN-and-CAT/blob/main/logo1.png" width=150 alt="CAT-YOLO">
 </div>
 
 
 ## Benchmark Results
 
 ### Mix Attention Block with ResNet-101 and WRN-18 on CIFAR-10
 
<table>
        <tr>
            <th>Model</th>
            <th>Params(M)</th>
            <th>Top-1 Error(%)</th>
        </tr>
        <tr>
            <th>ResNet-101 + Mix Attention</th>
            <th>49.21</th>
            <th>6.17</th>
        </tr>
        <tr>
            <th>WRN-18 + Mix Attention</th>
            <th>26.03</th>
            <th>4.92</th>
        </tr>
</table>
    
    
### Mix Attention Block with ResNet-101 and WRN-18 on CIFAR-100

<table>
        <tr>
            <th>Model</th>
            <th>Params(M)</th>
            <th>Top-1 Error(%)</th>
        </tr>
        <tr>
            <th>ResNet-101 + Mix Attention</th>
            <th>49.21</th>
            <th>23.19</th>
        </tr>
        <tr>
            <th>WRN-18 + Mix Attention</th>
            <th>26.03</th>
            <th>19.11</th>
        </tr>
</table>


### MANet on CIFAR-10

<table>
        <tr>
            <th>Model</th>
            <th>Params(M)</th>
            <th>Top-1 Accuracy(%)</th>
        </tr>
        <tr>
            <th>MANet-B</th>
            <th>69.3</th>
            <th>96.9</th>
        </tr>
        <tr>
            <th>MANet-S</th>
            <th>23.4</th>
            <th>94.7</th>
        </tr>
         <tr>
            <th>MANet-T</th>
            <th>4.3</th>
            <th>93.4</th>
        </tr>
</table>


### MANet on CIFAR-100

<table>
        <tr>
            <th>Model</th>
            <th>Params(M)</th>
            <th>Top-1 Accuracy(%)</th>
        </tr>
        <tr>
            <th>MANet-B</th>
            <th>69.3</th>
            <th>84.7</th>
        </tr>
        <tr>
            <th>MANet-S</th>
            <th>23.4</th>
            <th>79.8</th>
        </tr>
         <tr>
            <th>MANet-T</th>
            <th>4.3</th>
            <th>75.2</th>
        </tr>
</table>


### CAT-YOLO on COCO 2017
<table>
        <tr>
            <th>Model</th>
            <th>Backbone</th>
          <th>Params(M)</th>
          <th>Latency(ms)</th>
            <th>AP</th>
        </tr>
        <tr>
            <th>CAT-YOLO</th>
            <th>MANet-T</th>
            <th>6.17</th>
          <th>9.7(TITAN RTX)</th>
          <th>24.7</th>
        </tr>
</table>
