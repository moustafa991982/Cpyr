def get_vis(frame):
    new_frame = frame.type(torch.int).tolist()
    new_frame = [str(i) for i in new_frame]
    s = ''.join(new_frame)
    new_frame = s[-4:] + s[:8]
    
    return (int(new_frame,2) *0.01)-16

class Noise:
    def __init__(self,noise_type):
        self.noise_modes = ['None','1_pos_bias','0.5_pos_bias','0.1_pos_bias','1_neg_bias','0.5_neg_bias','0.1_neg_bias','high_square','moderate_square','low_square','normal','double','ten']
        self.noise_type = noise_type
        
        self.normal = normal.Normal(0, 0.032)
        self.double_normal = normal.Normal(0, 0.032*2)
        self.ten_normal = normal.Normal(0, 0.32)
        
        self.const_bias = 1
        
        t = np.linspace(0, 1, 400, endpoint=False)
        self.high_square = list((signal.square(2 * np.pi * 32 * t)+1)/2)
        self.moderate_square = list((signal.square(2 * np.pi * 16 * t)+1)/2)
        self.low_square = list((signal.square(2 * np.pi * 4 * t)+1)/2)
    def add_noise(self,value):
        if self.noise_type == 'None':
            return value 
        elif self.noise_type == '1_pos_bias':
            value = value + 1 
        elif self.noise_type == '0.5_pos_bias':
            value = value + 0.5
        elif self.noise_type == '0.1_pos_bias':
            value = value + 0.1

        elif self.noise_type == '1_neg_bias':
            value = value - 1
        elif self.noise_type == '0.5_neg_bias':
            value = value - 0.5
        elif self.noise_type == '0.1_neg_bias':
            value = value - 0.1
            
        elif self.noise_type == 'high_square':
            value = value * self.high_square.pop()
        elif self.noise_type == 'moderate_square':
            value = value * self.moderate_square.pop()
        elif self.noise_type == 'low_square':
            value = value * self.low_square.pop()
        elif self.noise_type == 'normal':
            value = value + self.normal.sample()
        elif self.noise_type == 'double':
            value = value + self.double_normal.sample()
        elif self.noise_type == 'ten':
            value = value + self.ten_normal.sample()
        return value
