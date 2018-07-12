from sklearn.metrics import mean_squared_error
from .models import make_model
import numpy as np
import math
from keras.optimizers import Adam
from keras.models import clone_model
import keras
import keras.backend as K

def fine_tuning_loss(y_true,y_pred):    #
    return K.mean(K.square(y_true[:,:,:,1]-(y_pred[:,:,:,0]*y_true[:,:,:,1]+y_pred[:,:,:,1])) + 2*y_pred[:,:,:,0]*K.square(y_true[:,:,:,2]) - K.square(y_true[:,:,:,2]))

sigma_arr = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
sigma_arr = sigma_arr*5

class Fine_tuning:

    def __init__(self, noisy_image, ep):
        
        self.noisy_img = np.float32(noisy_image)
        self.noisy_img /= 255.
        
        self.img_x = self.noisy_img.shape[0]
        self.img_y = self.noisy_img.shape[1]
        
        self.ep = ep
        self.mini_batch_size = 1
        
        self.model_copy = make_model(self.img_x, self.img_y)
        #self.model_copy.save_weights('./weights/sigma_estimation_model.hdf5')
    
    def preprocessing(self, sigma_hat):
        
        self.X_data = (self.noisy_img - 0.5) / 0.2
        self.X_data = self.X_data.reshape(1,self.img_x, self.img_y, 1)
        
        self.Y_data = np.zeros((1,self.img_x, self.img_y,3))
        #self.Y_data[:,:,:,0] = self.clean_img
        self.Y_data[:,:,:,1] = self.noisy_img
        self.Y_data[:,:,:,2] = sigma_hat/255.
    
    def get_model(self):
        
        model = clone_model(self.model_copy)
        model.load_weights('./weights/sigma_estimation_model.hdf5')
        adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        model.compile(loss=fine_tuning_loss, optimizer=adam)
        
        return model

    def estimation(self):
        
        min_sig_index = 0
        max_sig_index = 20
        
        while(True):

            save_result = Save_result()
            
            sigma_hat_index = (min_sig_index + max_sig_index)//2
            
            sigma_hat = sigma_arr[sigma_hat_index]
            self.preprocessing(sigma_hat)
            
            print ('current sigma_hat : ', sigma_hat)
            print ('')

            model = self.get_model()
            model.fit(self.X_data, self.Y_data, verbose=0, batch_size = self.mini_batch_size, epochs = self.ep,callbacks=[save_result])
            status = save_result.get_result()

            del model
            
            if status == True:
                max_sig_index = sigma_hat_index
            else:
                min_sig_index = sigma_hat_index

            if sigma_hat_index == (min_sig_index + max_sig_index)//2:
                sigma_hat = sigma_arr[sigma_hat_index]
                break
        

        return sigma_hat

class Save_result(keras.callbacks.Callback):
    def __init__(self):    
        self.loss_loss_than_zero = False
        return
    
    def on_train_begin(self, logs={}):
        return
        
    def on_train_end(self, logs={}):
        return
        
    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('loss')
        if current_loss < 0:
            print("Epoch %05d: early stopping, Loss < 0" % epoch)
            self.model.stop_training = True
            self.loss_loss_than_zero = True
        return
    
    def get_result(self):
        return self.loss_loss_than_zero
        

