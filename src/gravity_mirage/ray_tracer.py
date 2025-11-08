"""
Ray tracing para visualización de lensing gravitacional
"""
import numpy as np
from scipy.integrate import solve_ivp
from .physics import SchwarzschildBlackHole

class GravitationalRayTracer:
    """
    Implementa ray tracing considerando efectos gravitacionales
    """
    
    def __init__(self, black_hole: SchwarzschildBlackHole):
        self.bh = black_hole
        
    def trace_photon_simple(self, 
                           initial_pos: np.ndarray,
                           initial_dir: np.ndarray,
                           max_distance: float = 1000) -> np.ndarray:
        """
        Traza un fotón usando aproximación de campo débil (más rápido)
        
        Args:
            initial_pos: Posición inicial [x, y] (en unidades de Rs)
            initial_dir: Dirección inicial normalizada [dx, dy]
            max_distance: Distancia máxima a trazar
            
        Returns:
            Array de posiciones [N, 2]
        """
        pos = np.array(initial_pos, dtype=float)
        direction = np.array(initial_dir, dtype=float)
        direction = direction / np.linalg.norm(direction)
        
        positions = [pos.copy()]
        step_size = 0.1 * self.bh.schwarzschild_radius
        
        for _ in range(int(max_distance / step_size)):
            # Distancia al agujero negro
            r = np.linalg.norm(pos)
            
            # Si está muy cerca, es capturado
            if r < 1.1 * self.bh.schwarzschild_radius:
                break
            
            # Calcular deflexión
            # Componente radial hacia el agujero negro
            radial_dir = -pos / r
            
            # Fuerza de deflexión proporcional a 1/r²
            deflection_strength = (self.bh.schwarzschild_radius / r) ** 2
            deflection = radial_dir * deflection_strength * 0.1
            
            # Actualizar dirección
            direction = direction + deflection
            direction = direction / np.linalg.norm(direction)
            
            # Avanzar posición
            pos = pos + direction * step_size
            positions.append(pos.copy())
            
            # Si se aleja demasiado, terminar
            if r > max_distance:
                break
        
        return np.array(positions)
    
    def trace_photon_geodesic(self,
                             initial_pos_spherical: tuple[float, float, float],
                             initial_velocity: tuple[float, float, float],
                             lambda_max: float = 100) -> np.ndarray:
        """
        Traza un fotón resolviendo ecuaciones geodésicas completas
        (más preciso pero más lento)
        
        Args:
            initial_pos_spherical: (r, θ, φ) posición inicial
            initial_velocity: (dr/dλ, dθ/dλ, dφ/dλ) velocidades iniciales
            lambda_max: Parámetro afín máximo
            
        Returns:
            Solución de las ecuaciones geodésicas
        """
        r0, theta0, phi0 = initial_pos_spherical
        dr0, dtheta0, dphi0 = initial_velocity
        
        # Condición inicial para dt/dλ (de la condición de fotón: g_μν dx^μ dx^ν = 0)
        rs = self.bh.schwarzschild_radius
        f = 1 - rs / r0
        dt0 = np.sqrt((dr0 ** 2 / f + r0 ** 2 * (dtheta0 ** 2 + 
                      np.sin(theta0) ** 2 * dphi0 ** 2)) / f)
        
        # Estado inicial: [t, r, θ, φ, dt/dλ, dr/dλ, dθ/dλ, dφ/dλ]
        state0 = np.array([0, r0, theta0, phi0, dt0, dr0, dtheta0, dphi0])
        
        # Resolver ecuaciones geodésicas
        lambda_span = (0, lambda_max)
        lambda_eval = np.linspace(0, lambda_max, 1000)
        
        solution = solve_ivp(
            lambda l, s: self.bh.geodesic_equations(s, l),
            lambda_span,
            state0,
            t_eval=lambda_eval,
            method='RK45',
            rtol=1e-8
        )
        
        return solution