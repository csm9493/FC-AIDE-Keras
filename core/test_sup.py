from sklearn.metrics import mean_squared_error
from .models import make_model
import numpy as np
import math

from keras.optimizers import Adam
from skimage import measure

import scipy.io as sio

from keras import backend as K
def fine_tuning_loss(y_true,y_pred):    #
    return K.mean(K.square(y_true[:,:,:,1]-(y_pred[:,:,:,0]*y_true[:,:,:,1]+y_pred[:,:,:,1])) + 2*y_pred[:,:,:,0]*K.square(y_true[:,:,:,2]) - K.square(y_true[:,:,:,2]))

class Supervised_test:
    
    def __init__(self, clean_image, noisy_image, noise_sigma):
        
        self.clean_img = np.float32(clean_image)
        self.noisy_img = np.float32(noisy_image)
        self.noise_sigma = noise_sigma
        
        self.img_x = clean_image.shape[0]
        self.img_y = clean_image.shape[1]
        
        return

    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = measure.compare_ssim(X, X_hat, dynamic_range=X.max() - X.min())
        
        return test_SSIM
    
    def preprocessing(self):
        
        self.noisy_img /= 255.
        self.clean_img /= 255.
        
        self.X_data = (self.noisy_img - 0.5) / 0.2
        self.X_data = self.X_data.reshape(1,self.img_x, self.img_y, 1)
    
    def denoising(self):
        
        self.preprocessing()
        self.model = make_model(self.img_x, self.img_y)
        self.model.load_weights('./weights/' + 'sigma' + str(self.noise_sigma) + '.hdf5')

        returned_score = self.model.predict(self.X_data,batch_size=1, verbose=0)
        returned_score = np.array(returned_score)
        returned_score = returned_score.reshape(1,self.img_x,self.img_y,2)

        denoised_test_image = returned_score[0,:,:,0] * (self.noisy_img) + returned_score[0,:,:,1]

        ssim = self.get_SSIM(self.clean_img, denoised_test_image)
        psnr = self.get_PSNR(self.clean_img, denoised_test_image)


        return denoised_test_image, psnr, ssim