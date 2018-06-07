from core.test_sup import Supervised_test as test_sup
from scipy import misc
import numpy as np

file_name = 'cman'
file_name_clean = file_name + '_clean.png'
file_path = './data/'

noise_mean = 0
noise_sigma = 30

if noise_sigma not in [15, 25, 30, 50, 75]:
    print ('No weight file')
    exit()
    
clean_image = misc.imread(file_path + file_name_clean)
noisy_image = clean_image + np.random.normal(noise_mean, noise_sigma, clean_image.shape)

t_s = test_sup(clean_image, noisy_image, noise_sigma)
denoised_img, psnr, ssim = t_s.denoising()

misc.imsave(file_name +'_denoised_sup.png', denoised_img)

print ('PSNR : ' + str(round(psnr,2)) + '\nSSIM : ' + str(round(ssim,4)))