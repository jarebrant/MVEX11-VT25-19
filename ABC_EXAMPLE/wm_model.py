import numpy as np
import random
from variables import *
import os



bm_path = brainmask

tissue_grid_path = wm_mask_binarized

bm_slice = np.loadtxt(bm_path, delimiter=",")

tissue_grid = np.loadtxt(tissue_grid_path, delimiter=",")

# Basics


directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  

#Här skapar vi klasser, vilka fungerar som mallar för objekt. p_cell = proliferativ cell, m_cell = motil cell
class cell: 
    _id_counter = 1  

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = cell._id_counter
        cell._id_counter += 1 

class p_cell(cell):  
    def __init__(self, x, y):
        super().__init__(x, y)

class m_cell(cell):  
    def __init__(self, x, y):
        super().__init__(x, y)
        self.velocity = random.choice(directions)  # Eftersom propagationen är deterministic behöver vi en initial hastighet som den kommer att gå hela tiden om inte den får reorientation

# ---------------------------
# Skapa initial tumör
# ---------------------------
def create_initial_tumor(intensity_grid, R_eff_init, mitt_koord):
    cells = []
    num_rows, num_cols = intensity_grid.shape
    cx, cy = mitt_koord
    R = R_eff_init
    p_center, p_edge = 8/12, 0.181818181

    for x in range(num_rows):
        for y in range(num_cols):
            I = intensity_grid[x, y]
            total = int(round(I * 12))

            d = min(np.hypot(x-cx, y-cy), R)
            frac = (p_center + (p_edge-p_center)*(d/R)) if R>0 else p_center

            p_cells = int(round(total * frac))
            
            m_cells = 0
            if p_cells>8:
                m_cells+= 8-p_cells
            m_cells += total - p_cells
            if m_cells>4:
                p_cells += m_cells - 4
                
            p_cells = min(p_cells, 8)
            m_cells = min(m_cells, 4)

            cells += [p_cell(x, y)] * p_cells
            cells += [m_cell(x, y)] * m_cells

    return cells


# ---------------------------
# Funktioner för reorientation och propagation
# ---------------------------

def density_gradient(p_grid, m_grid, x, y, bm_slice,grid_size):
    #Beräknar den viktade densitetsgradienten 
    Gx, Gy = 0, 0  
    total_weight = 0  
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and bm_slice[nx, ny] == 1):
            weight = np.count_nonzero(p_grid[nx, ny]) + np.count_nonzero(m_grid[nx, ny])
            Gx += dx * weight  
            Gy += dy * weight  
            total_weight += weight #viktad denistet
    if total_weight > 0:
        Gx /= total_weight
        Gy /= total_weight
    return Gx, Gy


def node_flux(m_grid, x, y, bm_slice, grid_size):
    #Beräknar den genomsnittliga rörelsen för motila celler i en nod 
    Jx, Jy = 0, 0
    total_cells = np.count_nonzero(m_grid[x, y])  

    if total_cells == 0:
        return 0, 0  

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and bm_slice[nx, ny] == 1):
            Jx += dx
            Jy += dy

    return Jx / total_cells, Jy / total_cells  


def repulsive_reorientation(p_grid, m_grid, cell, alpha, bm_slice, grid_size):
    #Implementerar repulsiv reorientation enligt Tektonidis 
    Gx, Gy = density_gradient(p_grid, m_grid, cell.x, cell.y, bm_slice, grid_size)
    Jx, Jy = node_flux(m_grid, cell.x, cell.y, bm_slice, grid_size)
    dot_product = Gx * Jx + Gy * Jy  #Ekvation A.12 i tektonidis
    prob = np.exp(-alpha * dot_product)  

    if np.random.rand() < prob:
        weights = []
        for dx, dy in directions:
            nx, ny = cell.x + dx, cell.y + dy
            if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and bm_slice[nx, ny] == 1):
                neighbor_density = np.count_nonzero(m_grid[nx, ny])
                weights.append((np.exp(-alpha * neighbor_density), (dx, dy)))
        if weights:
            probabilities, moves = zip(*weights) #sannolikheterna som beräknas från A.12 och alla möjliga håll man kan ta är uppdelade.
            probabilities = np.array(probabilities) / np.sum(probabilities) #vi normaliserar
            best_dx, best_dy = moves[np.random.choice(len(moves), p=probabilities)] #Vi väljer en riktning baserad på sannolikheterna, ger oss bäst riktning
        else:
            best_dx, best_dy = 0, 0
        return best_dx, best_dy
    else:
        return cell.velocity  


def propagate(cell, grid_size):
    #Propagerar cellen i sin nuvarande hastighet, cellerna går åt samma riktning om inte de är reorienterade.
    dx, dy = cell.velocity
    new_x, new_y = cell.x + dx, cell.y + dy #går samma håll som innan

    if (0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1] and bm_slice[new_x, new_y] == 1):

        return new_x, new_y
    return cell.x, cell.y  # Behåll samma position om utanför gränsen

# ---------------------------
# Funktion som uppdaterar griden
# ---------------------------

def update_grid(grid_size, cells, p_grid, m_grid, p_node_capacity, m_node_capacity, pp, wm_migration_prob, gm_migration_prob, alpha, a_pm, b_pm, theta_pm, k_pm, a_mp, b_mp, theta_mp, k_mp, bm_slice, tissue_grid):


    new_cells = [] # Lista som sparar nya celler
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)] # Tillåtna riktningar för "proliferation"

    for cell in cells:  # Koden nedan itereras för varje cell i listan "cells"

        # ---------------------------
        # "Phenotype Swiching"
        # ---------------------------
    
        local_density = np.count_nonzero(p_grid[cell.x, cell.y]) + np.count_nonzero(m_grid[cell.x, cell.y]) 
        pm = a_pm + 0.5*(b_pm-a_pm)*(1+np.tanh(k_pm*(local_density-theta_pm))) # Sannolikheten att en cell övergår från proliferativ till motil
        mp = a_mp + 0.5*(b_mp-a_mp)*(1+np.tanh(k_mp*(local_density-theta_mp))) # Sannolikheten att en cell övergår från motil till proliferativ

        if np.random.rand() < pm:  # En cell övergår från proliferativ till motil enligt sannolikheten pm
            for channel in range(p_node_capacity):
                if p_grid[cell.x, cell.y, channel] == cell.id:
                    p_grid[cell.x, cell.y, channel] = 0
                    break
                
            for kanal in range(m_node_capacity):
                if m_grid[cell.x, cell.y, kanal] == 0:
                    m_grid[cell.x, cell.y, kanal] = cell.id
                    break
            cell.__class__ = m_cell
            cell.velocity = random.choice(directions)  # Ge den en initial velocity

        elif np.random.rand() < mp:  # En cell övergår från motil till proliferativ enligt sannolikheten mp
            for kanal in range(m_node_capacity):
                if m_grid[cell.x, cell.y, kanal] == cell.id:
                    m_grid[cell.x, cell.y, kanal] = 0
                    break
            for channel in range(p_node_capacity):
                if p_grid[cell.x, cell.y, channel] == 0:
                    p_grid[cell.x, cell.y, channel] = cell.id
                    break
            cell.__class__ = p_cell

        # ---------------------------
        # "Proliferation" 
        # ---------------------------

        if isinstance(cell, p_cell): # Om det finns en ledig kanal i någon av grann-noderna så tillåts proliferation med en viss sannolikhet, pp. 
            for dx, dy in neighbor_offsets:
                new_x, new_y = cell.x + dx, cell.y + dy
                if (0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1] and bm_slice[new_x, new_y] == 1):
                    occupancy = np.count_nonzero(p_grid[new_x, new_y])
                    if occupancy < p_node_capacity:
                        if np.random.rand() < pp:
                            new_cell = p_cell(new_x, new_y)
                            new_cells.append(new_cell)
                            for channel in range(p_node_capacity):
                                if p_grid[new_x, new_y, channel] == 0:
                                    p_grid[new_x, new_y, channel] = new_cell.id
                                    break

        # ---------------------------
        # "Reorientation" och "propagation"
        # ---------------------------
        elif isinstance(cell, m_cell):
            if tissue_grid[cell.x, cell.y] == True: 
               if np.random.rand() < wm_migration_prob:  
                   cell.x, cell.y = propagate(cell,grid_size)
                   dx, dy = repulsive_reorientation(p_grid, m_grid, cell, alpha, bm_slice, grid_size)
                   cell.velocity = (dx, dy)
            elif tissue_grid[cell.x, cell.y] == False:
                if np.random.rand() < gm_migration_prob:  
                   cell.x, cell.y = propagate(cell,grid_size)
                   dx, dy = repulsive_reorientation(p_grid, m_grid, cell, alpha,bm_slice, grid_size)
                   cell.velocity = (dx, dy)

    cells.extend(new_cells) 
    return cells, p_grid, m_grid


def simulate(pp, wm_migration_prob, gm_migration_prob, alpha, a_pm, b_pm, theta_pm, k_pm, a_mp, b_mp, theta_mp, k_mp):
    
    #Create initial cell grid based on mr_data
    matrix = np.loadtxt(start_intensity,delimiter=',')
    cells = create_initial_tumor(matrix,eff_radie, mid_point)
    grid_size = matrix.shape

    # Proliferative Grid
    p_node_capacity = 8
    p_grid = np.zeros((grid_size[0], grid_size[1], p_node_capacity), dtype=int)

    # Motile Grid
    m_node_capacity = 4  
    m_grid = np.zeros((grid_size[0], grid_size[1], m_node_capacity), dtype=int)


    for cell in cells:
        if isinstance(cell,p_cell):
            for channel in range(p_node_capacity):
                if p_grid[cell.x, cell.y, channel] == 0:
                    p_grid[cell.x, cell.y, channel] = cell.id
                    break
        elif isinstance(cell,m_cell):
            for channel in range(m_node_capacity):
                if m_grid[cell.x, cell.y, channel] == 0:
                    m_grid[cell.x, cell.y, channel] = cell.id
                    break
                
    for t in range(time_steps):
        cells, p_grid, m_grid = update_grid(grid_size, cells, p_grid, m_grid, p_node_capacity, m_node_capacity, pp, gm_migration_prob, wm_migration_prob, alpha, a_pm, b_pm, theta_pm, k_pm, a_mp, b_mp, theta_mp, k_mp, bm_slice, tissue_grid)

    density_grid = np.zeros((grid_size[0], grid_size[1]))
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            total_density = np.count_nonzero(p_grid[x, y]) + np.count_nonzero(m_grid[x, y])
            if total_density >= 1:
                density_grid[x, y] = total_density
            else:
                density_grid[x, y] = 0  
                

    norm_grid = (density_grid / 12 )
    
    return np.asarray(norm_grid, dtype=np.float32)



