import pygame
import numpy as np
import random
import heapq
from gesture_input import GestureController

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 700
GRID_SIZE = 40
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DVRP-MHSI Simulation")
clock = pygame.time.Clock()

# --- Parameters ---
NUM_AGENTS = 10
V_MAX = 3.0
K_ATTRACT = 1.5
K_OBS = 5000.0
K_SWARM = 800.0
D_SENSE = 150.0
D_SAFE = 50.0
DT = 0.1

# --- A* Algorithm ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, static_obstacles):
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        neighbor = (node[0] + dx, node[1] + dy)
        if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS:
            if neighbor not in static_obstacles:
                if dx != 0 and dy != 0:
                    if (node[0] + dx, node[1]) in static_obstacles or (node[0], node[1] + dy) in static_obstacles:
                        continue
                neighbors.append(neighbor)
    return neighbors

def astar(start, goal, static_obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for neighbor in get_neighbors(current, static_obstacles):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
    return []

# --- Agent Class ---
class Agent:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.path = []
        self.path_index = 0
    
    def update(self, target, obstacles, all_agents, static_obstacles):
        if not self.path or self.path_index >= len(self.path):
            start_grid = (int(self.pos[0] // GRID_SIZE), int(self.pos[1] // GRID_SIZE))
            goal_grid = (int(target[0] // GRID_SIZE), int(target[1] // GRID_SIZE))
            self.path = astar(start_grid, goal_grid, static_obstacles)
            self.path_index = 0

        if self.path and self.path_index < len(self.path):
            waypoint = np.array([self.path[self.path_index][0] * GRID_SIZE + GRID_SIZE//2,
                                 self.path[self.path_index][1] * GRID_SIZE + GRID_SIZE//2], dtype=float)
            if np.linalg.norm(self.pos - waypoint) < 15:
                self.path_index += 1
            target_for_force = waypoint
        else:
            target_for_force = target

        dist_to_target = np.linalg.norm(target_for_force - self.pos)
        if dist_to_target > 1.0:
            f_attract = K_ATTRACT * (target_for_force - self.pos) / dist_to_target
        else:
            f_attract = np.array([0.0, 0.0])
            
        f_obs = np.array([0.0, 0.0])
        for obs in obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if 0 < dist < D_SENSE:
                force_mag = K_OBS * (1.0/dist - 1.0/D_SENSE) / (dist**2)
                f_obs += force_mag * (self.pos - obs) / dist
                
        f_swarm = np.array([0.0, 0.0])
        for other in all_agents:
            if other.id != self.id:
                dist = np.linalg.norm(self.pos - other.pos)
                if 0 < dist < D_SAFE:
                    force_mag = K_SWARM * (1.0/dist - 1.0/D_SAFE) / (dist**2)
                    f_swarm += force_mag * (self.pos - other.pos) / dist

        self.vel = f_attract + f_obs + f_swarm
        speed = np.linalg.norm(self.vel)
        if speed > V_MAX:
            self.vel = (self.vel / speed) * V_MAX
            
        self.pos += self.vel * DT
        self.pos[0] = np.clip(self.pos[0], 20, WIDTH - 20)
        self.pos[1] = np.clip(self.pos[1], 20, HEIGHT - 20)

    def draw(self, surface):
        if self.path and self.path_index < len(self.path):
            for i in range(self.path_index, len(self.path)):
                pygame.draw.circle(surface, (50, 50, 100), 
                                   (self.path[i][0] * GRID_SIZE + GRID_SIZE//2,
                                    self.path[i][1] * GRID_SIZE + GRID_SIZE//2), 3)
        pygame.draw.circle(surface, (0, 200, 255), (int(self.pos[0]), int(self.pos[1])), 8)

# --- Setup Environment ---
agents = [Agent(i, random.randint(50, 200), random.randint(50, 650)) for i in range(NUM_AGENTS)]
target = np.array([850.0, 350.0])
obstacles = [np.array([500.0, 200.0]), np.array([500.0, 500.0])]

# --- STATIC OBSTACLES (Walls) ---
static_obstacles = set()
for i in range(0, ROWS):
    if i < 5 or i > 10:
        static_obstacles.add((15, i))

# --- Gesture Controller ---
gesture = GestureController(WIDTH, HEIGHT)
use_gesture = False

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            target = np.array(pygame.mouse.get_pos(), dtype=float)

    keys = pygame.key.get_pressed()
    
    # Toggle Gesture Mode with 'G'
    if keys[pygame.K_g]:
        use_gesture = not use_gesture
        pygame.time.wait(200) # Debounce
    
    if use_gesture:
        gesture_target = gesture.get_target_position()
        if gesture_target is not None:
            target = np.array(gesture_target, dtype=float)
    else:
        # Keyboard control
        move_step = 10.0
        if keys[pygame.K_UP] or keys[pygame.K_w]: target[1] -= move_step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: target[1] += move_step
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: target[0] -= move_step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: target[0] += move_step
    
    target[0] = np.clip(target[0], 20, WIDTH - 20)
    target[1] = np.clip(target[1], 20, HEIGHT - 20)

    t = pygame.time.get_ticks() / 1000.0
    obstacles[0][1] = 200 + 100 * np.sin(t)
    obstacles[1][1] = 500 + 100 * np.cos(t)

    for agent in agents:
        agent.update(target, obstacles, agents, static_obstacles)

    # --- Drawing ---
    screen.fill((20, 20, 30))
    
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, (30, 30, 45), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, (30, 30, 45), (0, y), (WIDTH, y))

    for obs in static_obstacles:
        pygame.draw.rect(screen, (100, 100, 100), 
                         (obs[0]*GRID_SIZE, obs[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

    for obs in obstacles:
        pygame.draw.circle(screen, (255, 50, 50), (int(obs[0]), int(obs[1])), 15)
        pygame.draw.circle(screen, (100, 0, 0), (int(obs[0]), int(obs[1])), int(D_SENSE), 1)
        
    pygame.draw.circle(screen, (50, 255, 50), (int(target[0]), int(target[1])), 10)
    
    for agent in agents:
        agent.draw(screen)

    pygame.display.flip()
    clock.tick(60)

gesture.release()
pygame.quit()