import numpy as np # type: ignore
import pandas as pd # type: ignore
import math 
from matplotlib.path import Path # type: ignore

from skimage import measure # type: ignore


class tilt_computer:

    def __init__(self, magnetogram, threshold, SC):
        self.magnetogram = magnetogram
        self.threshold = threshold
        self.SC = SC


    # contouring strong field regions to generate masks
    
    def contouring_strongfieldregions(self):

        contours_pos = measure.find_contours(self.magnetogram.data, self.threshold)
        contours_neg = measure.find_contours(-self.magnetogram.data, self.threshold)

        # building positive mask
        masks_pos = []
        mask_pos = np.zeros(self.magnetogram.data.shape, dtype = bool)

        for ii in range(len(contours_pos)):
            if len(contours_pos[ii]) > 0:
                path     = Path(contours_pos[ii])
                y, x     = np.meshgrid(np.arange(self.magnetogram.data.shape[0]), np.arange(self.magnetogram.data.shape[1]), indexing = 'ij')
                points   = np.vstack((y.ravel(), x.ravel())).T
                mask_pos = path.contains_points(points).reshape(self.magnetogram.data.shape)

                masks_pos.append(mask_pos)

        total_pos_mask = masks_pos[0]

        for jj in range(1, len(masks_pos)):
            total_pos_mask = np.logical_or(total_pos_mask, masks_pos[jj])

        self.total_pos_mask_disp = np.ones_like(total_pos_mask) * np.nan
        self.total_pos_mask_disp[total_pos_mask == True] = 1


        # building negative mask
        masks_neg = []
        mask_neg = np.zeros(self.magnetogram.data.shape, dtype = bool)

        for ii in range(len(contours_neg)):
            if len(contours_neg[ii]) > 0:
                path     = Path(contours_neg[ii])
                y, x     = np.meshgrid(np.arange(self.magnetogram.data.shape[0]), np.arange(self.magnetogram.data.shape[1]), indexing = 'ij')
                points   = np.vstack((y.ravel(), x.ravel())).T
                mask_neg = path.contains_points(points).reshape(self.magnetogram.data.shape)

                masks_neg.append(mask_neg)

        total_neg_mask = masks_neg[0]

        for jj in range(1, len(masks_neg)):
            total_neg_mask = np.logical_or(total_neg_mask, masks_neg[jj])

        self.total_neg_mask_disp = np.ones_like(total_neg_mask) * np.nan
        self.total_neg_mask_disp[total_neg_mask == True] = 1
        
        return [total_neg_mask, self.total_neg_mask_disp, total_pos_mask, self.total_pos_mask_disp]
    
    # finding flux-weighted centers for the positive and negative polarities

    def fluxweighted_center(self):
        B  = self.magnetogram.data
        nx = self.magnetogram.data.shape[1]   # number of columns (horizontal)
        ny = self.magnetogram.data.shape[0]   # number of rows (vertical)

        ##
        # positive polarity's flux weighted center

        dxBsum_pos = 0
        Bsum_x_pos = 0

        dyBsum_pos = 0
        Bsum_y_pos = 0

        x_fw_pos = 0
        y_fw_pos = 0

        for i in range(1, nx-1):   # horizontal
            for j in range(1, ny-1):   # vertical
                if np.isnan(self.total_pos_mask_disp[j, i]):
                    continue
                if np.isnan(B[j, i]):
                    continue

                dxBsum_pos += (0.5 + i) * B[j, i]
                Bsum_x_pos += B[j, i]

                dyBsum_pos += (0.5 + j) * B[j, i]
                Bsum_y_pos += B[j, i]

        # leftmost column
        i = 0
        for j in range(0, ny):
            if np.isnan(self.total_pos_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_pos += 0.5 * B[j, i]
            Bsum_x_pos += B[j, i]

        # rightmost column
        i = nx - 1
        for j in range(0, ny):
            if np.isnan(self.total_pos_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_pos += 0.5 * B[j, i]
            Bsum_x_pos += B[j, i]

        # topmost row
        j = 0
        for i in range(0, nx):
            if np.isnan(self.total_pos_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_pos += 0.5 * B[j, i]
            Bsum_x_pos += B[j, i]

        # bottom-most row
        j = ny - 1
        for i in range(0, nx):
            if np.isnan(self.total_pos_mask_disp[j ,i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_pos += 0.5 * B[j, i]
            Bsum_x_pos += B[j, i]

        x_fw_pos = dxBsum_pos / Bsum_x_pos
        y_fw_pos = dyBsum_pos / Bsum_y_pos   # be careful about the origin of the plot window



        ##
        # negative polarity's flux weighted center

        dxBsum_neg = 0
        Bsum_x_neg = 0

        dyBsum_neg = 0
        Bsum_y_neg = 0

        x_fw_neg = 0
        y_fw_neg = 0

        for i in range(1, nx-1):   # horizontal
            for j in range(1, ny-1):   # vertical
                if np.isnan(self.total_neg_mask_disp[j, i]):
                    continue
                if np.isnan(B[j, i]):
                    continue

                dxBsum_neg += (0.5 + i) * B[j, i]
                Bsum_x_neg += B[j, i]

                dyBsum_neg += (0.5 + j) * B[j, i]
                Bsum_y_neg += B[j, i]

        # leftmost column
        i = 0
        for j in range(0, ny):
            if np.isnan(self.total_neg_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_neg += 0.5 * B[j, i]
            Bsum_x_neg += B[j, i]

        # rightmost column
        i = nx - 1
        for j in range(0, ny):
            if np.isnan(self.total_neg_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_neg += 0.5 * B[j, i]
            Bsum_x_neg += B[j, i]

        # topmost row
        j = 0
        for i in range(0, nx):
            if np.isnan(self.total_neg_mask_disp[j, i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_neg += 0.5 * B[j, i]
            Bsum_x_neg += B[j, i]

        # bottom-most row
        j = ny - 1
        for i in range(0, nx):
            if np.isnan(self.total_neg_mask_disp[j ,i]):
                continue
            if np.isnan(B[j, i]):
                continue

            dxBsum_neg += 0.5 * B[j, i]
            Bsum_x_neg += B[j, i]

        x_fw_neg = dxBsum_neg / Bsum_x_neg
        y_fw_neg = dyBsum_neg / Bsum_y_neg   # be careful about the origin of the plot window

        self.x_fw, self.y_fw = [x_fw_pos, x_fw_neg], [y_fw_pos, y_fw_neg]
        
        return self.x_fw, self.y_fw
    
    # Computing tilt angle of the active region from the calculated flux weighted centers 
    def computeTilt(self):

        if self.magnetogram.header['LAT_FWT'] >= 0:
            hem = 'N'
        if self.magnetogram.header['LAT_FWT'] < 0:
            hem = 'S'
        
        radsindeg = np.pi / 180
        cdelt1_arcsec = (math.atan( (self.magnetogram.header['rsun_ref'] * self.magnetogram.header['cdelt1'] * radsindeg) / (self.magnetogram.header['dsun_obs']) )) * (1 / radsindeg) * 3600   # pixel lenght in arcsec as viewed by the observer
        self.dpix = cdelt1_arcsec * (self.magnetogram.header['rsun_ref'] / self.magnetogram.header['rsun_obs'])
        
        data = [self.magnetogram.header['HARPNUM'], self.magnetogram.header['NOAA_AR'], self.SC, hem, self.x_fw[1], self.y_fw[1], self.x_fw[0], self.y_fw[0], self.dpix, 'False', 'False', 0, 0]
        columns = ['HARPNUM', 'NOAA_AR', 'Solar Cycle', 'Hemisphere', 'x_fw_neg [px]', 'y_fw_neg [px]', 'x_fw_pos [px]', 'y_fw_pos [px]', 'dpix [m]', 'HALEness', 'JOYness', 'Absolute Tilt', 'Relative Tilt']

        self.df_main = pd.DataFrame(data = [data], columns = columns)

        HH = False
        JJ = False

        df = self.df_main.copy()

        ## Determining HALEness and JOYness

        # Odd solar cycle
        if np.mod(df['Solar Cycle'].iloc[0], 2) != 0:
            
            # Northern hemisphere
            if df['Hemisphere'].iloc[0] == 'N':
                
                if df['x_fw_pos [px]'].iloc[0] >= df['x_fw_neg [px]'].iloc[0]:
                    HH = True
                    
                    if df['y_fw_pos [px]'].iloc[0] >= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                else:                
                    if df['y_fw_pos [px]'].iloc[0] <= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                        
            # Southern hemisphere
            if df['Hemisphere'].iloc[0] == 'S':
                
                if df['x_fw_pos [px]'].iloc[0] <= df['x_fw_neg [px]'].iloc[0]:
                    HH = True

                    if df['y_fw_pos [px]'].iloc[0] >= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                
                else:                
                    if df['y_fw_pos [px]'].iloc[0] <= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True


        # Even solar cycle   
        elif np.mod(df['Solar Cycle'].iloc[0], 2) == 0:
            
            # Northern hemisphere
            if df['Hemisphere'].iloc[0] == 'N':
                
                if df['x_fw_pos [px]'].iloc[0] <= df['x_fw_neg [px]'].iloc[0]:
                    HH = True
                    
                    if df['y_fw_pos [px]'].iloc[0] <= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                    
                else:              
                    if df['y_fw_pos [px]'].iloc[0] >= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                    
            # Southern hemisphere
            if df['Hemisphere'].iloc[0] == 'S':
                
                if df['x_fw_pos [px]'].iloc[0] >= df['x_fw_neg [px]'].iloc[0]:
                    HH = True
                    
                    if df['y_fw_neg [px]'].iloc[0] >= df['y_fw_neg [px]'].iloc[0]:
                        JJ = True
                
                else:                
                    if df['y_fw_neg [px]'].iloc[0] <= df['y_fw_pos [px]'].iloc[0]:
                        JJ = True

        self.df_main.loc[0, 'HALEness'] = HH
        self.df_main.loc[0, 'JOYness']  = JJ


        # Tilt angle calculation

        dx_fw = np.abs(self.df_main['x_fw_neg [px]'].iloc[0] - self.df_main['x_fw_pos [px]'].iloc[0])
        dy_fw = np.abs(self.df_main['y_fw_neg [px]'].iloc[0] - self.df_main['y_fw_pos [px]'].iloc[0]) 

        alpha = math.atan(dy_fw / dx_fw) * (180 / np.pi)

        T_rel = 0
        T_abs = 0

        # Odd solar cycle
        if np.mod(df['Solar Cycle'].iloc[0], 2) != 0:
            
            # Northern hemisphere
            if df['Hemisphere'].iloc[0] == 'N':
                
                if JJ == True:

                    T_rel = 180 - alpha

                    if HH == True:
                        T_abs = -alpha

                    elif HH == False:
                        T_abs = T_rel

                elif JJ == False:

                    T_rel = -(180 - alpha)

                    if HH == True:
                        T_abs = alpha

                    elif HH == False:
                        T_abs = T_rel
            
            # Southern hemisphere
            elif df['Hemisphere'].iloc[0] == 'S':
                
                if JJ == True:
                    
                    T_rel = -(180 - alpha)
                    
                    if HH == True:
                        T_abs = T_rel
                    
                    elif HH == False:
                        T_abs = alpha
                
                elif JJ == False:
                    
                    T_rel = 180 - alpha
                    
                    if HH == True:
                        T_abs = T_rel
                    
                    elif HH == False:
                        T_abs = -alpha
        
        # Even solar cycle
        elif np.mod(df['Solar Cycle'].iloc[0], 2) == 0:
            
            # Northern hemisphere
            if df['Hemisphere'].iloc[0] == 'N':
                
                if JJ == True:
                    
                    T_rel = 180 - alpha
                    
                    if HH == True:
                        T_abs = T_rel
                    
                    elif HH == False:
                        T_abs = -alpha
                        
                elif JJ == False:
                    
                    T_rel = -(180 - alpha)
                    
                    if HH == True:
                        T_abs = T_rel
                    
                    elif HH == False:
                        T_abs = alpha
            
            # Southern hemisphere
            elif df['Hemisphere'].iloc[0] == 'S':
                
                if JJ == True:
                    
                    T_rel = -(180 - alpha)
                    
                    if HH == True:
                        T_abs = alpha
                        
                    elif HH == False:
                        T_abs = T_rel
                
                elif JJ == False:
                    
                    T_rel = 180 - alpha
                    
                    if HH == True:
                        T_abs = -alpha
                    
                    elif HH == False:
                        T_abs = T_rel

        self.df_main.loc[0, 'Absolute Tilt'] = T_abs
        self.df_main.loc[0, 'Relative Tilt'] = T_rel

        return self.df_main

