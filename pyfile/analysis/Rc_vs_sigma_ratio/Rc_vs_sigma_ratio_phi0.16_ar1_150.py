import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np

# boxsize=150
# phi_array = np.array([0.0005*4**j for j in range(5)])
# N_array = phi_array*(boxsize/0.5)**3/(4./3.*np.pi)

phi=0.16
ar=1./150.

sigma_ratio_array=np.array([[17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ],
 [17.72453851, 13.29340388, 10.63472311,  8.86226925,  7.59623079,
   6.64670194,  5.9081795 ,  5.31736155,  4.83396505,  4.43113463,
   4.09027812,  3.79811539,  3.5449077 ,  3.32335097,  3.12785974,
   2.95408975,  2.79861134,  2.65868078,  2.53207693,  2.41698252,
   2.31189633,  2.21556731,  2.12694462,  2.04513906,  2.0141521 ,
   2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ,  2.0141521 ]])

eta_array=np.array([[ 0.2       ,  0.26666667,  0.33333333,  0.4       ,  0.46666667,
   0.53333333,  0.6       ,  0.66666667,  0.73333333,  0.8       ,
   0.86666667,  0.93333333,  1.        ,  1.06666667,  1.13333333,
   1.2       ,  1.26666667,  1.33333333,  1.4       ,  1.46666667,
   1.53333333,  1.6       ,  1.66666667,  1.73333333,  1.76      ,
   1.76      ,  1.76      ,  1.76      ,  1.76      ,  1.76      ],
 [ 0.4       ,  0.53333333,  0.66666667,  0.8       ,  0.93333333,
   1.06666667,  1.2       ,  1.33333333,  1.46666667,  1.6       ,
   1.73333333,  1.86666667,  2.        ,  2.13333333,  2.26666667,
   2.4       ,  2.53333333,  2.66666667,  2.8       ,  2.93333333,
   3.06666667,  3.2       ,  3.33333333,  3.46666667,  3.52      ,
   3.52      ,  3.52      ,  3.52      ,  3.52      ,  3.52      ],
 [ 0.6       ,  0.8       ,  1.        ,  1.2       ,  1.4       ,
   1.6       ,  1.8       ,  2.        ,  2.2       ,  2.4       ,
   2.6       ,  2.8       ,  3.        ,  3.2       ,  3.4       ,
   3.6       ,  3.8       ,  4.        ,  4.2       ,  4.4       ,
   4.6       ,  4.8       ,  5.        ,  5.2       ,  5.28      ,
   5.28      ,  5.28      ,  5.28      ,  5.28      ,  5.28      ],
 [ 0.8       ,  1.06666667,  1.33333333,  1.6       ,  1.86666667,
   2.13333333,  2.4       ,  2.66666667,  2.93333333,  3.2       ,
   3.46666667,  3.73333333,  4.        ,  4.26666667,  4.53333333,
   4.8       ,  5.06666667,  5.33333333,  5.6       ,  5.86666667,
   6.13333333,  6.4       ,  6.66666667,  6.93333333,  7.04      ,
   7.04      ,  7.04      ,  7.04      ,  7.04      ,  7.04      ],
 [ 1.        ,  1.33333333,  1.66666667,  2.        ,  2.33333333,
   2.66666667,  3.        ,  3.33333333,  3.66666667,  4.        ,
   4.33333333,  4.66666667,  5.        ,  5.33333333,  5.66666667,
   6.        ,  6.33333333,  6.66666667,  7.        ,  7.33333333,
   7.66666667,  8.        ,  8.33333333,  8.66666667,  8.8       ,
   8.8       ,  8.8       ,  8.8       ,  8.8       ,  8.8       ],
 [ 1.2       ,  1.6       ,  2.        ,  2.4       ,  2.8       ,
   3.2       ,  3.6       ,  4.        ,  4.4       ,  4.8       ,
   5.2       ,  5.6       ,  6.        ,  6.4       ,  6.8       ,
   7.2       ,  7.6       ,  8.        ,  8.4       ,  8.8       ,
   9.2       ,  9.6       , 10.        , 10.4       , 10.56      ,
  10.56      , 10.56      , 10.56      , 10.56      , 10.56      ],
 [ 1.4       ,  1.86666667,  2.33333333,  2.8       ,  3.26666667,
   3.73333333,  4.2       ,  4.66666667,  5.13333333,  5.6       ,
   6.06666667,  6.53333333,  7.        ,  7.46666667,  7.93333333,
   8.4       ,  8.86666667,  9.33333333,  9.8       , 10.26666667,
  10.73333333, 11.2       , 11.66666667, 12.13333333, 12.32      ,
  12.32      , 12.32      , 12.32      , 12.32      , 12.32      ],
 [ 1.6       ,  2.13333333,  2.66666667,  3.2       ,  3.73333333,
   4.26666667,  4.8       ,  5.33333333,  5.86666667,  6.4       ,
   6.93333333,  7.46666667,  8.        ,  8.53333333,  9.06666667,
   9.6       , 10.13333333, 10.66666667, 11.2       , 11.73333333,
  12.26666667, 12.8       , 13.33333333, 13.86666667, 14.08      ,
  14.08      , 14.08      , 14.08      , 14.08      , 14.08      ],
 [ 1.8       ,  2.4       ,  3.        ,  3.6       ,  4.2       ,
   4.8       ,  5.4       ,  6.        ,  6.6       ,  7.2       ,
   7.8       ,  8.4       ,  9.        ,  9.6       , 10.2       ,
  10.8       , 11.4       , 12.        , 12.6       , 13.2       ,
  13.8       , 14.4       , 15.        , 15.6       , 15.84      ,
  15.84      , 15.84      , 15.84      , 15.84      , 15.84      ],
 [ 2.        ,  2.66666667,  3.33333333,  4.        ,  4.66666667,
   5.33333333,  6.        ,  6.66666667,  7.33333333,  8.        ,
   8.66666667,  9.33333333, 10.        , 10.66666667, 11.33333333,
  12.        , 12.66666667, 13.33333333, 14.        , 14.66666667,
  15.33333333, 16.        , 16.66666667, 17.33333333, 17.6       ,
  17.6       , 17.6       , 17.6       , 17.6       , 17.6       ],
 [ 2.2       ,  2.93333333,  3.66666667,  4.4       ,  5.13333333,
   5.86666667,  6.6       ,  7.33333333,  8.06666667,  8.8       ,
   9.53333333, 10.26666667, 11.        , 11.73333333, 12.46666667,
  13.2       , 13.93333333, 14.66666667, 15.4       , 16.13333333,
  16.86666667, 17.6       , 18.33333333, 19.06666667, 19.36      ,
  19.36      , 19.36      , 19.36      , 19.36      , 19.36      ],
 [ 2.4       ,  3.2       ,  4.        ,  4.8       ,  5.6       ,
   6.4       ,  7.2       ,  8.        ,  8.8       ,  9.6       ,
  10.4       , 11.2       , 12.        , 12.8       , 13.6       ,
  14.4       , 15.2       , 16.        , 16.8       , 17.6       ,
  18.4       , 19.2       , 20.        , 20.8       , 21.12      ,
  21.12      , 21.12      , 21.12      , 21.12      , 21.12      ],
 [ 2.6       ,  3.46666667,  4.33333333,  5.2       ,  6.06666667,
   6.93333333,  7.8       ,  8.66666667,  9.53333333, 10.4       ,
  11.26666667, 12.13333333, 13.        , 13.86666667, 14.73333333,
  15.6       , 16.46666667, 17.33333333, 18.2       , 19.06666667,
  19.93333333, 20.8       , 21.66666667, 22.53333333, 22.88      ,
  22.88      , 22.88      , 22.88      , 22.88      , 22.88      ],
 [ 2.8       ,  3.73333333,  4.66666667,  5.6       ,  6.53333333,
   7.46666667,  8.4       ,  9.33333333, 10.26666667, 11.2       ,
  12.13333333, 13.06666667, 14.        , 14.93333333, 15.86666667,
  16.8       , 17.73333333, 18.66666667, 19.6       , 20.53333333,
  21.46666667, 22.4       , 23.33333333, 24.26666667, 24.64      ,
  24.64      , 24.64      , 24.64      , 24.64      , 24.64      ],
 [ 3.        ,  4.        ,  5.        ,  6.        ,  7.        ,
   8.        ,  9.        , 10.        , 11.        , 12.        ,
  13.        , 14.        , 15.        , 16.        , 17.        ,
  18.        , 19.        , 20.        , 21.        , 22.        ,
  23.        , 24.        , 25.        , 26.        , 26.4       ,
  26.4       , 26.4       , 26.4       , 26.4       , 26.4       ],
 [ 3.2       ,  4.26666667,  5.33333333,  6.4       ,  7.46666667,
   8.53333333,  9.6       , 10.66666667, 11.73333333, 12.8       ,
  13.86666667, 14.93333333, 16.        , 17.06666667, 18.13333333,
  19.2       , 20.26666667, 21.33333333, 22.4       , 23.46666667,
  24.53333333, 25.6       , 26.66666667, 27.73333333, 28.16      ,
  28.16      , 28.16      , 28.16      , 28.16      , 28.16      ],
 [ 3.4       ,  4.53333333,  5.66666667,  6.8       ,  7.93333333,
   9.06666667, 10.2       , 11.33333333, 12.46666667, 13.6       ,
  14.73333333, 15.86666667, 17.        , 18.13333333, 19.26666667,
  20.4       , 21.53333333, 22.66666667, 23.8       , 24.93333333,
  26.06666667, 27.2       , 28.33333333, 29.46666667, 29.92      ,
  29.92      , 29.92      , 29.92      , 29.92      , 29.92      ],
 [ 3.6       ,  4.8       ,  6.        ,  7.2       ,  8.4       ,
   9.6       , 10.8       , 12.        , 13.2       , 14.4       ,
  15.6       , 16.8       , 18.        , 19.2       , 20.4       ,
  21.6       , 22.8       , 24.        , 25.2       , 26.4       ,
  27.6       , 28.8       , 30.        , 31.2       , 31.68      ,
  31.68      , 31.68      , 31.68      , 31.68      , 31.68      ],
 [ 3.8       ,  5.06666667,  6.33333333,  7.6       ,  8.86666667,
  10.13333333, 11.4       , 12.66666667, 13.93333333, 15.2       ,
  16.46666667, 17.73333333, 19.        , 20.26666667, 21.53333333,
  22.8       , 24.06666667, 25.33333333, 26.6       , 27.86666667,
  29.13333333, 30.4       , 31.66666667, 32.93333333, 33.44      ,
  33.44      , 33.44      , 33.44      , 33.44      , 33.44      ],
 [ 4.        ,  5.33333333,  6.66666667,  8.        ,  9.33333333,
  10.66666667, 12.        , 13.33333333, 14.66666667, 16.        ,
  17.33333333, 18.66666667, 20.        , 21.33333333, 22.66666667,
  24.        , 25.33333333, 26.66666667, 28.        , 29.33333333,
  30.66666667, 32.        , 33.33333333, 34.66666667, 35.2       ,
  35.2       , 35.2       , 35.2       , 35.2       , 35.2       ],
 [ 4.2       ,  5.6       ,  7.        ,  8.4       ,  9.8       ,
  11.2       , 12.6       , 14.        , 15.4       , 16.8       ,
  18.2       , 19.6       , 21.        , 22.4       , 23.8       ,
  25.2       , 26.6       , 28.        , 29.4       , 30.8       ,
  32.2       , 33.6       , 35.        , 36.4       , 36.96      ,
  36.96      , 36.96      , 36.96      , 36.96      , 36.96      ],
 [ 4.4       ,  5.86666667,  7.33333333,  8.8       , 10.26666667,
  11.73333333, 13.2       , 14.66666667, 16.13333333, 17.6       ,
  19.06666667, 20.53333333, 22.        , 23.46666667, 24.93333333,
  26.4       , 27.86666667, 29.33333333, 30.8       , 32.26666667,
  33.73333333, 35.2       , 36.66666667, 38.13333333, 38.72      ,
  38.72      , 38.72      , 38.72      , 38.72      , 38.72      ],
 [ 4.6       ,  6.13333333,  7.66666667,  9.2       , 10.73333333,
  12.26666667, 13.8       , 15.33333333, 16.86666667, 18.4       ,
  19.93333333, 21.46666667, 23.        , 24.53333333, 26.06666667,
  27.6       , 29.13333333, 30.66666667, 32.2       , 33.73333333,
  35.26666667, 36.8       , 38.33333333, 39.86666667, 40.48      ,
  40.48      , 40.48      , 40.48      , 40.48      , 40.48      ]])

error_array=np.array([[1.37711364e-01, 1.12152484e-01, 9.38226818e-02, 7.97688869e-02,
  6.85960293e-02, 5.95249421e-02, 5.20436388e-02, 4.57921960e-02,
  4.05116467e-02, 3.60084582e-02, 3.21340925e-02, 2.87642643e-02,
  2.57979250e-02, 2.31549682e-02, 2.07751712e-02, 1.86169891e-02,
  1.66476724e-02, 1.48459476e-02, 1.31961439e-02, 1.16844968e-02,
  1.03008598e-02, 9.03661921e-03, 7.88465835e-03, 6.83974080e-03,
  6.45065394e-03, 6.45065394e-03, 6.45065394e-03, 6.45065394e-03,
  6.45065394e-03, 6.45065394e-03],
 [1.17706246e-01, 9.10021530e-02, 7.25836964e-02, 5.90842929e-02,
  4.88937803e-02, 4.10071946e-02, 3.47284212e-02, 2.96150724e-02,
  2.53667298e-02, 2.17670854e-02, 1.86669366e-02, 1.59607812e-02,
  1.35729940e-02, 1.14501370e-02, 9.55979262e-03, 7.88300831e-03,
  6.41029274e-03, 5.13518723e-03, 4.04977324e-03, 3.14281469e-03,
  2.39974559e-03, 1.80333076e-03, 1.33462824e-03, 9.74198396e-04,
  8.56030194e-04, 8.56030194e-04, 8.56030194e-04, 8.56030194e-04,
  8.56030194e-04, 8.56030194e-04],
 [9.64683645e-02, 7.03117400e-02, 5.33853992e-02, 4.16761098e-02,
  3.30428295e-02, 2.63012166e-02, 2.07796841e-02, 1.61431494e-02,
  1.22379496e-02, 8.99625711e-03, 6.37863896e-03, 4.34471031e-03,
  2.83353182e-03, 1.76554692e-03, 1.05161483e-03, 6.04932583e-04,
  3.48205509e-04, 2.15087624e-04, 1.48985267e-04, 1.10534209e-04,
  8.22222029e-05, 5.92701234e-05, 4.10086702e-05, 2.72282180e-05,
  2.28591633e-05, 2.28591633e-05, 2.28591633e-05, 2.28591633e-05,
  2.28591633e-05, 2.28591633e-05],
 [7.93166013e-02, 5.54227254e-02, 4.04123778e-02, 2.97040816e-02,
  2.14150028e-02, 1.48573576e-02, 9.78168114e-03, 6.04690638e-03,
  3.47308291e-03, 1.83315456e-03, 8.82028727e-04, 3.97893690e-04,
  2.02693714e-04, 1.40188468e-04, 1.04871479e-04, 7.28960967e-05,
  4.63757334e-05, 2.72783378e-05, 1.49853750e-05, 7.74544827e-06,
  3.78621583e-06, 1.75685911e-06, 7.75781599e-07, 3.26566223e-07,
  2.28022212e-07, 2.28022212e-07, 2.28022212e-07, 2.28022212e-07,
  2.28022212e-07, 2.28022212e-07],
 [6.70785214e-02, 4.48917251e-02, 3.04079647e-02, 1.98297113e-02,
  1.20791194e-02, 6.73470327e-03, 3.37259721e-03, 1.47873889e-03,
  5.57236071e-04, 2.17288944e-04, 1.42862848e-04, 1.04215055e-04,
  6.55714176e-05, 3.59753744e-05, 1.76515741e-05, 7.87817826e-06,
  3.23303790e-06, 1.22860072e-06, 4.34409416e-07, 1.43367000e-07,
  4.42619138e-08, 1.28015883e-08, 3.47193309e-09, 8.83519524e-10,
  5.02079565e-10, 5.02079565e-10, 5.02079564e-10, 5.02079565e-10,
  5.02079564e-10, 5.02079565e-10],
 [5.92155131e-02, 3.76894806e-02, 2.33188630e-02, 1.32113124e-02,
  6.61231992e-03, 2.81769673e-03, 9.65387388e-04, 2.77292392e-04,
  1.62207650e-04, 1.18376133e-04, 6.78001001e-05, 3.20625209e-05,
  1.30542049e-05, 4.68019193e-06, 1.49668730e-06, 4.30343075e-07,
  1.11840552e-07, 2.63664731e-08, 5.65329275e-09, 1.10457131e-09,
  1.96951683e-10, 3.20833608e-11, 4.77896789e-12, 6.51315278e-13,
  2.86311975e-13, 2.86312645e-13, 2.86312756e-13, 2.86312297e-13,
  2.86311932e-13, 2.86312270e-13],
 [4.98292770e-02, 2.84610496e-02, 1.45837452e-02, 6.27028453e-03,
  2.10456332e-03, 4.99534215e-04, 1.85751110e-04, 1.33742247e-04,
  6.84017197e-05, 2.69666032e-05, 8.74379632e-06, 2.40653620e-06,
  5.72187606e-07, 1.18788162e-07, 2.16852453e-08, 3.49763499e-09,
  5.00116055e-10, 6.35403357e-11, 7.18464348e-12, 7.23773892e-13,
  6.50044282e-14, 5.23077511e-15, 5.51923885e-16, 3.88403317e-16,
  4.67180954e-16, 4.66594539e-16, 4.67474026e-16, 4.67733185e-16,
  4.71611759e-16, 4.70012725e-16],
 [4.23221668e-02, 2.12206993e-02, 8.78222206e-03, 2.74499304e-03,
  5.73285022e-04, 1.92008862e-04, 1.24447837e-04, 5.41121717e-05,
  1.76288446e-05, 4.60060002e-06, 9.91219481e-07, 1.79078882e-07,
  2.73587847e-08, 3.55193627e-09, 3.93078225e-10, 3.71596648e-11,
  3.00560329e-12, 2.08251826e-13, 1.23837654e-14, 7.72999400e-16,
  3.70917870e-16, 3.68054721e-16, 3.65513533e-16, 3.81041404e-16,
  4.65697748e-16, 4.65439330e-16, 4.61658282e-16, 4.66620591e-16,
  4.66242570e-16, 4.66270706e-16],
 [3.57453390e-02, 1.53595020e-02, 4.94008192e-03, 1.00827495e-03,
  2.12531394e-04, 1.40478954e-04, 5.54031480e-05, 1.51024240e-05,
  3.11684060e-06, 5.06131954e-07, 6.59435236e-08, 6.97117263e-09,
  6.02027775e-10, 4.26500514e-11, 2.48524230e-12, 1.19335296e-13,
  4.74354554e-15, 4.63382718e-16, 4.35244882e-16, 4.09201888e-16,
  3.63872182e-16, 3.58485679e-16, 3.59751392e-16, 3.81201580e-16,
  4.61352918e-16, 4.63107049e-16, 4.64226666e-16, 4.62073060e-16,
  4.59663439e-16, 4.60569353e-16],
 [2.87329361e-02, 9.87899025e-03, 2.15996145e-03, 2.75541538e-04,
  1.66723032e-04, 6.22954302e-05, 1.44428626e-05, 2.37333733e-06,
  2.91209361e-07, 2.73819507e-08, 2.00245330e-09, 1.14946003e-10,
  5.20876671e-12, 1.86982817e-13, 5.34631832e-15, 5.55211783e-16,
  4.59832609e-16, 4.19684883e-16, 4.30007382e-16, 4.01283139e-16,
  3.52315429e-16, 3.66102301e-16, 3.59725217e-16, 3.74461097e-16,
  4.62013550e-16, 4.63938073e-16, 4.63152370e-16, 4.66374944e-16,
  4.69415330e-16, 4.61655170e-16],
 [2.28676521e-02, 6.14138983e-03, 8.47903460e-04, 2.13116920e-04,
  9.62608725e-05, 2.23222205e-05, 3.35537449e-06, 3.53458389e-07,
  2.69593972e-08, 1.51363155e-09, 6.31530113e-11, 1.96980404e-12,
  4.61027592e-14, 9.80778764e-16, 4.79771993e-16, 5.24628483e-16,
  4.44847862e-16, 4.07527469e-16, 4.18056423e-16, 3.95703593e-16,
  3.52774591e-16, 3.55368479e-16, 3.46794495e-16, 3.66746732e-16,
  4.55907643e-16, 4.51967315e-16, 4.56088257e-16, 4.52148069e-16,
  4.61229206e-16, 4.52394409e-16],
 [1.76201171e-02, 3.51408958e-03, 3.17384161e-04, 1.63351610e-04,
  4.36494682e-05, 6.43633684e-06, 6.07790428e-07, 3.87512243e-08,
  1.70937758e-09, 5.28626358e-11, 1.15533296e-12, 1.79366640e-14,
  6.04621767e-16, 5.22539518e-16, 4.75895094e-16, 5.20672900e-16,
  4.55832480e-16, 4.16975747e-16, 4.18146572e-16, 3.98376009e-16,
  3.52844173e-16, 3.50700854e-16, 3.47606898e-16, 3.68631447e-16,
  4.55320078e-16, 4.56369798e-16, 4.58381481e-16, 4.51895528e-16,
  4.54700889e-16, 4.52917869e-16],
 [1.29130041e-02, 1.71714856e-03, 2.38373524e-04, 9.44208775e-05,
  1.50730445e-05, 1.35285638e-06, 7.53244298e-08, 2.70878869e-09,
  6.42088311e-11, 1.01531145e-12, 1.07898672e-14, 5.94784080e-16,
  5.51592391e-16, 5.02895876e-16, 4.68049272e-16, 5.12298549e-16,
  4.41656882e-16, 3.99895165e-16, 4.04512103e-16, 3.93898996e-16,
  3.41698081e-16, 3.47265164e-16, 3.42395031e-16, 3.68724615e-16,
  4.51335007e-16, 4.59430502e-16, 4.58932322e-16, 4.54796467e-16,
  4.54490634e-16, 4.58355537e-16],
 [9.07357180e-03, 7.30455414e-04, 2.00668681e-04, 4.49665280e-05,
  4.48103222e-06, 2.49032492e-07, 8.28245528e-09, 1.69755768e-10,
  2.17593662e-12, 1.75829878e-14, 6.93313074e-16, 5.88482821e-16,
  5.49180769e-16, 5.04775068e-16, 4.61667323e-16, 5.10109603e-16,
  4.46412132e-16, 3.97125707e-16, 4.07374716e-16, 3.89781126e-16,
  3.41896504e-16, 3.45799952e-16, 3.40767518e-16, 3.69744323e-16,
  4.52308262e-16, 4.53194907e-16, 4.56401260e-16, 4.56066844e-16,
  4.54814287e-16, 4.56802391e-16],
 [6.35513605e-03, 3.48978495e-04, 1.44108797e-04, 2.04048775e-05,
  1.31352950e-06, 4.50766170e-08, 8.70058438e-10, 9.68102707e-12,
  6.29376209e-14, 7.72558380e-16, 6.91559782e-16, 5.91027416e-16,
  5.59559057e-16, 5.05491908e-16, 4.62386086e-16, 5.14320748e-16,
  4.44237280e-16, 3.95194315e-16, 4.04050875e-16, 3.90892728e-16,
  3.45050319e-16, 3.38375169e-16, 3.46159296e-16, 3.70920267e-16,
  4.55273166e-16, 4.53921579e-16, 4.57094773e-16, 4.49730233e-16,
  4.54023092e-16, 4.55843056e-16],
 [4.03657989e-03, 2.69421356e-04, 8.52689249e-05, 7.38314924e-06,
  2.86811995e-07, 5.63111560e-09, 5.84626500e-11, 3.28261842e-13,
  1.28332092e-15, 6.99313814e-16, 6.56586790e-16, 5.65599222e-16,
  5.35711568e-16, 4.89108084e-16, 4.42309868e-16, 4.97596309e-16,
  4.30839487e-16, 3.84744324e-16, 3.97579363e-16, 3.79407697e-16,
  3.30968244e-16, 3.38973852e-16, 3.40780136e-16, 3.59119051e-16,
  4.49526545e-16, 4.50004511e-16, 4.49436596e-16, 4.58357359e-16,
  4.54266565e-16, 4.49190645e-16],
 [2.44533405e-03, 2.48229820e-04, 4.56554349e-05, 2.50739976e-06,
  5.98264957e-08, 6.81455644e-10, 3.84529493e-12, 1.09677465e-14,
  7.71234406e-16, 6.99838590e-16, 6.52407127e-16, 5.56119745e-16,
  5.30424000e-16, 4.84908212e-16, 4.41238137e-16, 5.03042953e-16,
  4.29747054e-16, 3.91378936e-16, 4.02554914e-16, 3.76783989e-16,
  3.30129536e-16, 3.47207190e-16, 3.40044212e-16, 3.63608758e-16,
  4.49312874e-16, 4.51728784e-16, 4.54841938e-16, 4.54285885e-16,
  4.49416752e-16, 4.50939817e-16],
 [1.36764911e-03, 2.00442506e-04, 2.21287656e-05, 7.63203857e-07,
  1.08418192e-08, 6.84838168e-11, 1.98586669e-13, 9.19011656e-16,
  7.75851700e-16, 6.98643198e-16, 6.48541764e-16, 5.60910144e-16,
  5.28003596e-16, 4.80886048e-16, 4.39881726e-16, 5.04328211e-16,
  4.33767652e-16, 3.85939251e-16, 3.93264042e-16, 3.80838389e-16,
  3.34291884e-16, 3.41502068e-16, 3.33813912e-16, 3.57022258e-16,
  4.51191182e-16, 4.50627246e-16, 4.47569756e-16, 4.52102580e-16,
  4.57642976e-16, 4.47206518e-16],
 [7.12170878e-04, 1.44841338e-04, 9.84939110e-06, 2.07575734e-07,
  1.68196306e-09, 5.59930018e-12, 7.95242666e-15, 8.13301925e-16,
  7.25589473e-16, 6.51661080e-16, 6.24513683e-16, 5.26683095e-16,
  5.06656505e-16, 4.57119773e-16, 4.20672560e-16, 4.86887436e-16,
  4.18671839e-16, 3.75944005e-16, 3.85762608e-16, 3.67521569e-16,
  3.22042144e-16, 3.32918341e-16, 3.29711892e-16, 3.55020417e-16,
  4.44001445e-16, 4.47848592e-16, 4.44073325e-16, 4.43163362e-16,
  4.52232865e-16, 4.49230236e-16],
 [3.84190520e-04, 9.45215445e-05, 4.01419517e-06, 5.15658138e-08,
  2.38464663e-10, 4.20486997e-13, 1.07289501e-15, 8.18778800e-16,
  7.22757639e-16, 6.49690020e-16, 6.20757627e-16, 5.30063745e-16,
  5.01956934e-16, 4.57941633e-16, 4.24398167e-16, 4.85658424e-16,
  4.10543376e-16, 3.79424594e-16, 3.85420989e-16, 3.68725185e-16,
  3.19352230e-16, 3.30609035e-16, 3.33624507e-16, 3.54292584e-16,
  4.48844174e-16, 4.41837616e-16, 4.53555879e-16, 4.40704944e-16,
  4.42561946e-16, 4.43992386e-16],
 [2.97421923e-04, 5.81846890e-05, 1.55723276e-06, 1.19319266e-08,
  3.02290106e-11, 2.66134634e-14, 1.03707673e-15, 8.14009035e-16,
  7.20072069e-16, 6.49983759e-16, 6.23203937e-16, 5.32884229e-16,
  5.05337169e-16, 4.58732721e-16, 4.19354969e-16, 4.86010149e-16,
  4.08215923e-16, 3.77322062e-16, 3.84709151e-16, 3.73744848e-16,
  3.19804664e-16, 3.29270656e-16, 3.30201747e-16, 3.56760314e-16,
  4.49830666e-16, 4.53048120e-16, 4.46632227e-16, 4.50865782e-16,
  4.50607903e-16, 4.43182548e-16],
 [2.85656857e-04, 3.34637623e-05, 5.55519003e-07, 2.46406502e-09,
  3.30491081e-12, 1.85601407e-15, 1.03705033e-15, 8.17196616e-16,
  7.24056537e-16, 6.53571905e-16, 6.27443668e-16, 5.27127198e-16,
  5.10461119e-16, 4.57397885e-16, 4.19872431e-16, 4.78922420e-16,
  4.11903767e-16, 3.73768557e-16, 3.83905667e-16, 3.72868349e-16,
  3.24354369e-16, 3.33827393e-16, 3.35405998e-16, 3.60336221e-16,
  4.46741353e-16, 4.44128450e-16, 4.41224481e-16, 4.46778248e-16,
  4.42959316e-16, 4.49139958e-16],
 [2.62850179e-04, 1.79639980e-05, 1.82949933e-07, 4.61702527e-10,
  3.22240323e-13, 1.18028133e-15, 1.03632543e-15, 8.17965660e-16,
  7.19885880e-16, 6.48433120e-16, 6.17114713e-16, 5.30734864e-16,
  5.01922717e-16, 4.57166311e-16, 4.23336697e-16, 4.82926212e-16,
  4.09874137e-16, 3.77401544e-16, 3.84176582e-16, 3.80408552e-16,
  3.21340099e-16, 3.32094558e-16, 3.25214721e-16, 3.61654440e-16,
  4.45696728e-16, 4.43566007e-16, 4.47890736e-16, 4.42918914e-16,
  4.48791169e-16, 4.45721414e-16]])



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
levels = np.logspace(-16, -2, 15)
k_array = np.zeros(np.shape(levels))
print(np.shape(levels)[0])
print(levels)

cs = ax.contourf(sigma_ratio_array, eta_array*sigma_ratio_array, error_array,
                levels, cmap=plt.cm.bone, norm=mpl.colors.LogNorm())
cs2 = ax.contour(cs,
                levels, norm=mpl.colors.LogNorm())

# #####Fitting
from scipy.optimize import curve_fit
def prop(x, k):
    return k*x

ydata = eta_array*sigma_ratio_array*0.5/np.sqrt(np.pi)
for i in range(1, len(levels)):    
    xdata = cs2.collections[i].get_paths()[0].vertices[:, 0]
    ydata = cs2.collections[i].get_paths()[0].vertices[:, 1]
    popt, pcov = curve_fit(prop, xdata, ydata)
    k_array[i] = popt
    fit_x = np.linspace(2, 12, 100)
    fit_y = prop(fit_x, popt)
    # ax.plot(fit_x, fit_y, linestyle='dashed')
    # ax.scatter(xdata, ydata, marker='+')
print(k_array)
# #####Fitting


ax.clabel(cs2, inline=True, fontsize=10, fmt='%2.1E', colors=[plt.cm.bone((20*i + 100)%291) for i in range(np.shape(levels)[0])])
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('Linear velocity % error')
cbar.ax.set_yscale('log')
# cbar.add_lines(cs)
# adding title and labels
ax.set_ylabel(r"$R_c/\sigma$")
ax.set_xlabel(r"$\Sigma/\sigma$")
ax.set_xlim((2., 12))
# ax.set_ylim((1, 20))
# ax.set_title(r"$R_c/\sigma$ vs. $\Sigma/\sigma$")
plt.savefig(f'img/rc_vs_sigmaratio_phi{phi}_ar{ar:.4f}.pdf', bbox_inches = 'tight', format='pdf')
plt.show()
