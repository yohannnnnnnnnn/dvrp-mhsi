import pygame
import numpy as np
import random

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DVRP-MHSI Simulation")
clock = pygame.time.Clock()

# --- Parameters from the Paper (via NotebookLM) ---
NUM_AGENTS = 10
V_MAX = 2.5            # Max speed
K_ATTRACT = 1.0        # Attraction strength to target
K_OBS = 5000.0         # Repulsion strength from obstacles
K_SWARM = 800.0        # Repulsion strength from other agents
D_SENSE = 150.0        # Obstacle detection range
D_SAFE = 50.0          # Swarm safety range
DT = 0.1               # Time step

# --- Agent Class ---
class Agent:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
    
    def update(self, target, obstacles, all_agents):
        # 1. Attractive Force (Guiding Field)
        dist_to_target = np.linalg.norm(target - self.pos)
        if dist_to_target > 1.0:
            f_attract = K_ATTRACT * (target - self.pos) / dist_to_target
        else:
            f_attract = np.array([0.0, 0.0])
            
        # 2. Obstacle Repulsion
        f_obs = np.array([0.0, 0.0])
        for obs in obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if 0 < dist < D_SENSE:
                # Formula from NotebookLM Step 3
                force_mag = K_OBS * (1.0/dist - 1.0/D_SENSE) / (dist**2)
                f_obs += force_mag * (self.pos - obs) / dist
                
        # 3. Swarm Repulsion (Collision Avoidance)
        f_swarm = np.array([0.0, 0.0])
        for other in all_agents:
            if other.id != self.id:
                dist = np.linalg.norm(self.pos - other.pos)
                if 0 < dist < D_SAFE:
                    force_mag = K_SWARM * (1.0/dist - 1.0/D_SAFE) / (dist**2)
                    f_swarm += force_mag * (self.pos - other.pos) / dist

        # 4. Combine Forces
        self.vel = f_attract + f_obs + f_swarm
        
        # Clamp velocity to V_MAX
        speed = np.linalg.norm(self.vel)
        if speed > V_MAX:
            self.vel = (self.vel / speed) * V_MAX
            
        # 5. Update Position
        self.pos += self.vel * DT
        
        # Keep agents within bounds
        self.pos[0] = np.clip(self.pos[0], 20, WIDTH - 20)
        self.pos[1] = np.clip(self.pos[1], 20, HEIGHT - 20)

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 200, 255), (int(self.pos[0]), int(self.pos[1])), 8)
        # Draw velocity vector (optional, for debugging)
        # pygame.draw.line(surface, (255, 255, 255), self.pos, self.pos + self.vel * 10, 2)

# --- Setup Environment ---
agents = [Agent(i, random.randint(100, 300), random.randint(100, 600)) for i in range(NUM_AGENTS)]
target = np.array([800.0, 350.0])
obstacles = [np.array([500.0, 200.0]), np.array([500.0, 500.0])]

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Mouse click sets new target
        if event.type == pygame.MOUSEBUTTONDOWN:
            target = np.array(pygame.mouse.get_pos(), dtype=float)

    # --- Keyboard Control for Target (Simulates Human Gesture) ---
    keys = pygame.key.get_pressed()
    move_step = 6.0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        target[1] -= move_step
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        target[1] += move_step
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        target[0] -= move_step
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        target[0] += move_step
    
    # Keep target within screen bounds
    target[0] = np.clip(target[0], 20, WIDTH - 20)
    target[1] = np.clip(target[1], 20, HEIGHT - 20)

    # Move obstacles dynamically (simulate dynamic obstacles)
    t = pygame.time.get_ticks() / 1000.0
    obstacles[0][1] = 200 + 100 * np.sin(t)
    obstacles[1][1] = 500 + 100 * np.cos(t)

    # Update agents
    for agent in agents:
        agent.update(target, obstacles, agents)

    # --- Drawing ---
    screen.fill((20, 20, 30))
    
    # Draw obstacles (Red)
    for obs in obstacles:
        pygame.draw.circle(screen, (255, 50, 50), (int(obs[0]), int(obs[1])), 15)
        pygame.draw.circle(screen, (100, 0, 0), (int(obs[0]), int(obs[1])), int(D_SENSE), 1)
        
    # Draw target (Green)
    pygame.draw.circle(screen, (50, 255, 50), (int(target[0]), int(target[1])), 10)
    
    # Draw agents
    for agent in agents:
        agent.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()