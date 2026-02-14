"""
Vehicle Telematics Data Generator
Generates synthetic vehicle telematics data for energy consumption anomaly detection.
"""

import numpy as np
import pandas as pd


class TelematicsDataGenerator:
    """
    Generate synthetic vehicle telematics data with energy consumption patterns.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_normal_data(self, n_samples=1000):
        """
        Generate normal (non-anomalous) vehicle telematics data.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        data : pandas DataFrame
            Telematics data with features
        """
        # Speed (km/h) - normal distribution around 60 km/h
        speed = np.random.normal(60, 15, n_samples)
        speed = np.clip(speed, 0, 120)
        
        # Acceleration (m/s²) - small values for normal driving
        acceleration = np.random.normal(0, 0.5, n_samples)
        acceleration = np.clip(acceleration, -2, 2)
        
        # RPM (revolutions per minute) - correlated with speed
        rpm = 1000 + speed * 25 + np.random.normal(0, 200, n_samples)
        rpm = np.clip(rpm, 600, 4000)
        
        # Engine load (%) - correlated with acceleration and speed
        engine_load = 30 + speed * 0.3 + acceleration * 5 + np.random.normal(0, 5, n_samples)
        engine_load = np.clip(engine_load, 0, 100)
        
        # Throttle position (%) - correlated with acceleration
        throttle = 20 + acceleration * 10 + np.random.normal(0, 5, n_samples)
        throttle = np.clip(throttle, 0, 100)
        
        # Brake pressure (bar) - inverse to throttle
        brake = np.maximum(0, 5 - acceleration * 3 + np.random.normal(0, 1, n_samples))
        brake = np.clip(brake, 0, 10)
        
        # Temperature (°C) - engine temperature
        temperature = np.random.normal(90, 5, n_samples)
        temperature = np.clip(temperature, 70, 110)
        
        # Fuel rate (L/h) - function of speed, acceleration, and engine load
        fuel_rate = 2 + speed * 0.05 + engine_load * 0.08 + np.abs(acceleration) * 0.5
        fuel_rate += np.random.normal(0, 0.5, n_samples)
        fuel_rate = np.clip(fuel_rate, 0, 20)
        
        # Energy consumption (kWh/km) - normal pattern
        energy = 0.15 + speed * 0.001 + engine_load * 0.002 + np.abs(acceleration) * 0.01
        energy += np.random.normal(0, 0.02, n_samples)
        energy = np.clip(energy, 0.05, 0.5)
        
        data = pd.DataFrame({
            'speed': speed,
            'acceleration': acceleration,
            'rpm': rpm,
            'engine_load': engine_load,
            'throttle': throttle,
            'brake': brake,
            'temperature': temperature,
            'fuel_rate': fuel_rate,
            'energy_consumption': energy
        })
        
        return data
    
    def generate_anomalous_data(self, n_samples=200):
        """
        Generate anomalous vehicle telematics data with unusual energy consumption.
        
        Parameters:
        -----------
        n_samples : int
            Number of anomalous samples to generate
            
        Returns:
        --------
        data : pandas DataFrame
            Anomalous telematics data
        """
        # Start with normal-ish data
        data = self.generate_normal_data(n_samples)
        
        # Create different types of anomalies
        n_per_type = n_samples // 4
        
        # Type 1: High energy consumption at low speed (inefficient driving)
        idx_start = 0
        idx_end = n_per_type
        data.loc[idx_start:idx_end-1, 'speed'] = np.random.uniform(20, 40, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'energy_consumption'] = np.random.uniform(0.35, 0.6, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'engine_load'] = np.random.uniform(60, 90, idx_end - idx_start)
        
        # Type 2: Excessive acceleration/braking cycles
        idx_start = n_per_type
        idx_end = 2 * n_per_type
        data.loc[idx_start:idx_end-1, 'acceleration'] = np.random.uniform(1.5, 3, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'brake'] = np.random.uniform(5, 10, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'energy_consumption'] = np.random.uniform(0.4, 0.7, idx_end - idx_start)
        
        # Type 3: High RPM with low efficiency
        idx_start = 2 * n_per_type
        idx_end = 3 * n_per_type
        data.loc[idx_start:idx_end-1, 'rpm'] = np.random.uniform(3500, 5000, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'throttle'] = np.random.uniform(70, 100, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'energy_consumption'] = np.random.uniform(0.45, 0.8, idx_end - idx_start)
        
        # Type 4: Temperature anomalies affecting efficiency
        idx_start = 3 * n_per_type
        idx_end = n_samples
        data.loc[idx_start:idx_end-1, 'temperature'] = np.random.uniform(110, 130, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'energy_consumption'] = np.random.uniform(0.4, 0.65, idx_end - idx_start)
        data.loc[idx_start:idx_end-1, 'engine_load'] = np.random.uniform(70, 95, idx_end - idx_start)
        
        return data
    
    def generate_dataset(self, n_normal=1000, n_anomalous=200):
        """
        Generate complete dataset with both normal and anomalous samples.
        
        Parameters:
        -----------
        n_normal : int
            Number of normal samples
        n_anomalous : int
            Number of anomalous samples
            
        Returns:
        --------
        X : pandas DataFrame
            Feature matrix
        y : numpy array
            Labels (0 = normal, 1 = anomalous)
        """
        # Generate normal and anomalous data
        normal_data = self.generate_normal_data(n_normal)
        anomalous_data = self.generate_anomalous_data(n_anomalous)
        
        # Combine data
        X = pd.concat([normal_data, anomalous_data], ignore_index=True)
        
        # Create labels
        y = np.concatenate([
            np.zeros(n_normal),
            np.ones(n_anomalous)
        ])
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(len(X))
        X = X.iloc[shuffle_idx].reset_index(drop=True)
        y = y[shuffle_idx]
        
        return X, y
