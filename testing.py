import numpy as np
import copy
import functions as f



test = np.ones(5, dtype=np.uint64, order='C')
liste = []
liste.append(copy.deepcopy(test))
liste.append(copy.deepcopy(test*4))
liste.append(copy.deepcopy(test*5))
liste.append(copy.deepcopy(test*7))
liste, tolerance = f.minimalizeAndCalcTolerance(liste, 2, 0)
print(str(liste))
print(str(tolerance))
