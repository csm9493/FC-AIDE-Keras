from core.sigma_est import Fine_tuning as sigma_est
from core.test_blind_ft import Fine_tuning as test_ft
from scipy import misc
import numpy as np

file_name = 'cman'
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

noisy_image = clean_image + np.random.normal(noise_mean, noise_sigma, clean_image.shape)

t_ft = test_ft(clean_image, noisy_image, sigma_hat)
denoised_img, psnr, ssim = t_ft.fine_tuning()

misc.imsave(file_name +'_denoised_blind_estimated_sigma_ft.png', denoised_img)

print ('PSNR : ' + str(round(psnr,2)) + '\nSSIM : ' + str(round(ssim,4)))