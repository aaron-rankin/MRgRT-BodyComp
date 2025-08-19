"""
Collection of transforms for simulating an observer delineation
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.signal import fftconvolve

class ObserverSimulator():

    def __init__(self, prob_bias=0.5, 
                 bias_scale=3, bias_stdev=1,
                 noise_scale=40, noise_stdev=5,
                 smoothing_sigma=6):
        self.prob_bias = prob_bias

        self.bias_scale = bias_scale
        self.bias_stdev = bias_stdev

        self.noise_strength = np.random.normal(loc=noise_scale, scale=noise_stdev)

        self.smoothing_sigma = smoothing_sigma

    def __call__(self, mask):
        # Get signed distance map of contour 
        # < 0 inside mask; > 0 outside 
        distance_map = self.signed_distance(mask)
        
        # Sample bias from normal dist.
        if np.random.random_sample() < self.prob_bias:
            bias = np.random.normal(loc=self.bias_scale, scale=self.bias_stdev)
        else:
            bias = np.random.normal(loc=0, scale=1)

        # Randomly flip sign of bias with prob 0.5
        bias = bias * -1 if np.random.random_sample() > 0.5 else bias

        simulated_map = self.simulate(distance_map, bias)
        output_mask = np.where(simulated_map <= 0, 1, 0) # Mask negative values and border
        return output_mask

    @staticmethod
    def signed_distance(mask):
        """
        Expects binary mask 
        """
        mask = np.round(mask).astype(bool)
        erode_mask = binary_erosion(mask, iterations=1).astype(bool)
        contour = np.logical_xor(mask, erode_mask).astype(int) # Extract contour
        dist = distance_transform_edt(1 - contour) # Get distance transform
        overlay = np.where(mask == 1, -dist, dist) # Flip sign inside mask
        return overlay
    
    def simulate(self, distance_map, bias, eps = 1e-24):
        """
        Main method - simulates random observer given distance map and bias
        """
        # Generate Gaussian noise
        noise = np.random.normal(loc=0, scale=1, size=(512, 512))
        noisy_map = self.noise_strength * noise

        #1-D  Gaussian
        t = np.linspace(-10, 10, 30) # Define kernel resolution? 
        bump = np.exp(-t**2 / (2 * (self.smoothing_sigma + eps)**2))
        bump /= np.trapz(bump)  # normalize the integral to 1
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :] # make a 2-D kernel out of it
        conv_noise = fftconvolve(noisy_map, kernel, mode='same') # Convolve noise map with smoothing kernel
        return distance_map + bias + conv_noise