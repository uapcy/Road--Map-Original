# seismic_utils.py - NEW MODULE for seismic-specific calculations

import numpy as np

def calculate_seismic_impedance(density, seismic_velocity):
    """
    Calculate acoustic impedance for seismic waves.
    Z = ρ × V, where ρ is density and V is seismic velocity.
    """
    return density * seismic_velocity

def calculate_reflection_coefficient(z1, z2):
    """
    Calculate seismic reflection coefficient at boundary between two materials.
    R = (Z2 - Z1) / (Z2 + Z1)
    """
    return (z2 - z1) / (z2 + z1)

def estimate_material_velocity(material_type):
    """
    Estimate typical seismic velocities for common materials.
    Returns velocity in m/s.
    """
    velocities = {
        'air': 330,
        'water': 1480,
        'sand': 300-1500,
        'clay': 1000-2500,
        'limestone': 2000-4500,
        'sandstone': 1500-4000,
        'granite': 4500-6000,
        'basalt': 4500-6500
    }
    return velocities.get(material_type.lower(), 3000)

def non_linear_oscillator_model(displacement, elastic_constant, mass, damping_coeff=0.05):
    """
    Implement the non-linear oscillator model from paper equations 13-20.
    Models the coupling between transverse and longitudinal oscillations.
    """
    L0 = 1.0
    L = 1.1
    
    r = np.abs(displacement)
    term1 = -4 * elastic_constant * displacement * (L - L0) / L
    term2 = -8 * elastic_constant * L0 * (r**3 / L**3 - r**5 / L**5)
    force = term1 + term2
    
    acceleration = force / mass - damping_coeff * np.gradient(displacement)
    
    return acceleration

def calculate_standing_wave_frequencies(length, seismic_velocity, n_modes=5):
    """
    Calculate standing wave frequencies in a structure of given length.
    f_n = (n × V) / (2 × L) for n = 1,2,3,...
    """
    frequencies = []
    for n in range(1, n_modes + 1):
        freq = (n * seismic_velocity) / (2 * length)
        frequencies.append(freq)
    return frequencies