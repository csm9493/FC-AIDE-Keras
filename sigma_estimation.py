from core.sigma_est import Fine_tuning as sigma_est
from scipy import misc
import numpy as np

file_name = 'barbara'
file_name_clean = file_name + '_clean.png'
file_path = './data/'

noise_mean = 0
noise_sigma = 30

epochs = 100

clean_image = misc.imread(file_path + file_name_clean)
noisy_image = clean_image + np.random.normal(noise_mean, noise_sigma, clean_image.shape)

s_est = sigma_est(noisy_image, epochs)
sigma_hat = s_est.estimation()

print ('true sigma : ' + str(noise_sigma))
print ('estimated sigma : ' + str(sigma_hat))