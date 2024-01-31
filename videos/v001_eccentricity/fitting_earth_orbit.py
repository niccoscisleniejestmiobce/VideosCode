import math
import numpy as np
from scipy.optimize import minimize

def ellipse_segment_area(alpha_start, alpha_stop, a, e, n):
    d_alpha = (alpha_stop-alpha_start)/n
    d_alpha_radians = math.radians(d_alpha)
    sin_d_alpha = math.sin(d_alpha_radians)
    
    alphas = np.linspace(math.radians(alpha_start), math.radians(alpha_stop), n+1)
    r = a*(1-e*e)/(1+e*np.cos(alphas))
    
    r1 = r[:-1]
    r2 = r[1:]

    return np.sum(0.5*sin_d_alpha*r1*r2)

def calc_error(t1, t2, t3, t4, e, offset):
    offset = -offset 
    t_sum = t1+t2+t3+t4
    
    t1_fraq = t1/t_sum
    t2_fraq = t2/t_sum
    t3_fraq = t3/t_sum
    t4_fraq = t4/t_sum
    
    full_ellipse_area = math.pi*math.sqrt(1-e*e)
    
    segment1 = ellipse_segment_area(offset, offset+90, 1, e, 800)/full_ellipse_area
    segment2 = ellipse_segment_area(offset+90, offset+180, 1, e, 800)/full_ellipse_area
    segment3 = ellipse_segment_area(offset+180, offset+270, 1, e, 800)/full_ellipse_area
    segment4 = ellipse_segment_area(offset+270, offset+360, 1, e, 800)/full_ellipse_area
    
    error = 0
    error += (segment1-t1_fraq)**2
    error += (segment2-t2_fraq)**2
    error += (segment3-t3_fraq)**2
    error += (segment4-t4_fraq)**2
    
    return error
    
def fit_ellipse(t1, t2, t3, t4):
    def error_to_min(args):
        e, offset = args
        return calc_error(t1, t2, t3, t4, e, offset)
    
    res = minimize(error_to_min, 
                   x0=(0.0, 10), 
                   method='SLSQP', 
                   tol=1e-20, 
                   bounds=((0.0001, 0.99999), (-180, 180)))

    print(res.message)
    
    return res.x
    
def format_degrees_as_h_m(angle):
    if angle < 0:
        angle += 360
    total_hours = angle / 360 * 24
    
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    
    return f"{hours}h {minutes}m"

def main():
    # season durations taken from Eugeniusz Rybka "Astronomia Ogólna"
    e_fitted, offset_fitted = fit_ellipse(92+21/24, # spring
                                          93+14/24, # summer
                                          89+18/24, # autumn
                                          89+1/24) # winter

    print(f"e = {e_fitted:.5f}, offset = {format_degrees_as_h_m(offset_fitted)}")

if __name__ == "__main__":
    main()
