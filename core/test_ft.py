from .models import make_model
import numpy as np
import math
import scipy.io as sio

from sklearn.metrics import mean_squared_error
from skimage import measure

ft_epoch_arr = {'sigma15':27, 'sigma25':16, 'sigma30':12, 'sigma50':9, 'sigma75':7}

class Fine_tuning:
    
    def __init__(self, clean_image, noisy_image, noise_sigma):
        
        self.clean_img = np.float32(clean_image)
        self.noisy_img = np.float32(noisy_image)
        self.noise_sigma = noise_sigma
        
        self.img_x = clean_image.shape[0]
        self.img_y = clean_image.shape[1]
        
        self.ep = ft_epoch_arr['sigma'+str(self.noise_sigma)]
        self.mini_batch_size = 1
        
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
        
        self.Y_data = np.zeros((1,self.img_x, self.img_y,3))
        self.Y_data[:,:,:,0] = self.clean_img
        self.Y_data[:,:,:,1] = self.noisy_img
        self.Y_data[:,:,:,2] = self.noise_sigma/255.

    def generate_flipped_image_set(self, X_data):
        
        if X_data.shape[3] == 1:
        
            flipped_image_set = []

            lr_flip = np.fliplr(X_data.reshape(self.img_x,self.img_y))
            ud_flip = np.flipud(X_data.reshape(self.img_x,self.img_y))
            lr_ud_flip = np.flipud(lr_flip)

            flipped_image_set = X_data.reshape(1,self.img_x,self.img_y,X_data.shape[3])
            flipped_image_set = np.vstack((flipped_image_set, lr_flip.reshape(1,self.img_x,self.img_y,X_data.shape[3])))
            flipped_image_set = np.vstack((flipped_image_set, ud_flip.reshape(1,self.img_x,self.img_y,X_data.shape[3])))
            flipped_image_set = np.vstack((flipped_image_set, lr_ud_flip.reshape(1,self.img_x,self.img_y,X_data.shape[3])))
            
        else:
            
            flipped_image_set = np.zeros((4,X_data.shape[1],X_data.shape[2],X_data.shape[3]))
            
            for i in range(3):
                
                origin = X_data[0,:,:,i]
                lr_flip = np.fliplr(X_data[0,:,:,i])
                ud_flip = np.flipud(X_data[0,:,:,i])
                lr_ud_flip = np.flipud(np.fliplr(X_data[0,:,:,i]))

                flipped_image_set[0,:,:,i] = origin
                flipped_image_set[1,:,:,i] = lr_flip
                flipped_image_set[2,:,:,i] = ud_flip
                flipped_image_set[3,:,:,i] = lr_ud_flip

        return flipped_image_set
    
    def reverse_flipped_image_set(self,X_data):
        
        origin_image = X_data[0]
        reverse_lr_flip = np.fliplr(X_data[1])
        reverse_ud_flip = np.flipud(X_data[2])
        reverse_lr_ud_flip = np.flipud(np.fliplr(X_data[3]))
        
        ensemble_image = (origin_image + reverse_lr_flip + reverse_ud_flip + reverse_lr_ud_flip)/4
    
        return ensemble_image
        
    def denoising(self):
        
        Z_data_flip = self.noisy_img.reshape(1,self.img_x,self.img_y,1)
        Z_data_flip = self.generate_flipped_image_set(Z_data_flip)
        
        returned_score = self.model.predict(self.X_data_flip,batch_size=4, verbose=0)
        returned_score = np.array(returned_score)
        returned_score = returned_score.reshape(4,self.img_x,self.img_y,2)

        denoised_test_image = returned_score[:,:,:,0] * (Z_data_flip[:,:,:,0]) + returned_score[:,:,:,1]
        denoised_test_image = np.clip(denoised_test_image, 0, 1)

        denoised_test_image = self.reverse_flipped_image_set(denoised_test_image)
        
        PSNR = self.get_PSNR(self.clean_img,denoised_test_image)
        SSIM = self.get_SSIM(self.clean_img,denoised_test_image)
        

        return denoised_test_image, PSNR, SSIM
    
    def fine_tuning(self):
        
        self.preprocessing()

        self.X_data_flip = self.X_data
        self.X_data_flip = self.generate_flipped_image_set(self.X_data)

        Y_data = self.Y_data
        Y_data = self.generate_flipped_image_set(Y_data)

        self.model = make_model(self.img_x, self.img_y)
        self.model.load_weights('./weights/' + 'sigma' + str(self.noise_sigma) + '.hdf5')

        self.model.fit(self.X_data_flip, Y_data, verbose=0, batch_size = self.mini_batch_size, epochs = self.ep)
        
        denoised_test_image, PSNR, SSIM = self.denoising()
        
        return denoised_test_image, PSNR, SSIM

            
        