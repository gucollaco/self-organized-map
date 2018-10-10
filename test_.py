# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:03:48 2018

@author: gusta
"""

import operator
import numpy as np
a = [1, 1.23, 1.21]

index, value = max(enumerate(a), key=operator.itemgetter(1))

print(np.argmin(a), np.amin(a))

print(int( 12/ 5), 12/5, 12%5)