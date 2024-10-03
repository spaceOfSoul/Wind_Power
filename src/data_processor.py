import warnings
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from windpowerlib.wind_speed import logarithmic_profile
from sklearnex import patch_sklearn, config_context
patch_sklearn()

from sklearn.base import BaseEstimator, TransformerMixin

# make data transformer
class UVTransformer(BaseEstimator, TransformerMixin):
    """Convert U, V wind component to WindSpeed, WindDirection by sklearn style.

        Parameters:
        u_feature_name (str): u component feature name
        v_feature_name (str): v component feature name
    """
    def __init__(self, u_feature_name:str, v_feature_name:str):
        self.u = u_feature_name
        self.v = v_feature_name
        
    def fit(self, X, y=None):
        """Take u, v components data

        Parameters:
        X (pd.DataFrame): DataFrame that contains u, v component features
        """
        if not all(feature in X.columns for feature in [self.u, self.v]):
            raise ValueError(f"'{self.u}' or '{self.v}' is not in the features of X")

        self.u_ws = X[self.u].to_numpy()
        self.v_ws = X[self.v].to_numpy()

        return self

    def transform(self, X, y=None):
        """Transform u,v components to wind speed and meteorological degree.
        NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv

        Parameters:
        X (pd.DataFrame): DataFrame that contains u, v component features

        Returns:
        X (pd.DataFrame): DataFrame with converted wind speed and direction
        """
        warnings.filterwarnings("ignore")

        # NOTE: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
        wind_speed = np.nansum([self.u_ws**2, self.v_ws**2], axis=0)**(1/2.)

        # math degree
        wind_direction = np.rad2deg(np.arctan2(self.v_ws, self.u_ws+1e-8))
        wind_direction[wind_direction < 0] += 360

        # meteorological degree
        wind_direction = 270 - wind_direction
        wind_direction[wind_direction < 0] += 360

        X['wind_speed'] = wind_speed
        X['wind_direction'] = wind_direction

        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
class WindTransformer(BaseEstimator, TransformerMixin):
    """Convert WindSpeed to hub height Windspeed by sklearn style.

        Parameters:
        wind_speed_feature_name (str): windspeed feature name
        wind_speed_height (int): height of the wind speed
        hub_height (int): height of the target wind speed
        roughness_length (int,float,str) : roughness_length of the surface, can be constant or feature name
    """
    def __init__(self, 
                 windspeed_feature_name:str,
                 wind_speed_height:int,
                 hub_height:int,
                 roughness_length):
        
        self.windspeed_str = windspeed_feature_name
        self.ref_height = wind_speed_height
        self.hub_height = hub_height
        self.rough = roughness_length
        
    def fit(self,
            X: pd.DataFrame,
            y=None):
        """Take u, v components data

        Parameters:
        X (pd.DataFrame): DataFrame that contains windspeed features
        """
        if not self.windspeed_str in X.columns:
            raise ValueError(f"'{self.windspeed_str}' is not in the features of X")

        self.windspeed = X[self.windspeed_str]

        if isinstance(self.rough, str):
            self.roughness = X[self.rough]

        elif isinstance(self.rough, (int, float, np.int64, np.int32, np.float64, np.float32)):
            self.roughness = self.rough
            
        else:
            raise ValueError("Invalid type for 'rough'. Expected str, int, or float.")

        return self

    def transform(self, X, y=None):
        """Transform windspeed to hub height by logarithmic wind profile.

        Parameters:
        X (pd.DataFrame): DataFrame that contains windspeed
        Returns:
        X (pd.DataFrame): DataFrame with converted wind speed
        """
        warnings.filterwarnings("ignore")
        X[f'wind_speed_{self.hub_height}m'] = logarithmic_profile(self.windspeed, 
                                                                  self.ref_height, 
                                                                  self.hub_height, # 소재지표고에 따라 변하게 해야할것같음, 다만 이 함수에 series로 넣으면 에러가 남.
                                                                  self.roughness)

        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)