import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5 * np.pi, 0.1)
y_cos = np.cos(x)
y_sin = np.sin(x)
plt.plot(x, y_cos)
plt.plot(x, y_sin)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Title')
plt.legend(['cosine', 'sine'])
plt.show()
