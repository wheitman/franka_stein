import imageio as iio
import matplotlib.pyplot as plt

camera = iio.get_reader("<video4>")
screenshot = camera.get_data(0)
camera.close()

plt.imshow(screenshot)
plt.show()
