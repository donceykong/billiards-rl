'''
This code supports the game engine and functionality for the agent to learn.
Decision Making Under Uncertainty Final Project (Spring 2023)

Author: Doncey Albin

Original script taken from: https://github.com/russs123/pool_tutorial
Video Tutorial : https://www.youtube.com/watch?v=txcOqDhrwBo
'''

import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
import random
pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 678
BOTTOM_PANEL = 50

#game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + BOTTOM_PANEL))
pygame.display.set_caption("Pool")

#pymunk space
space = pymunk.Space()
static_body = space.static_body
draw_options = pymunk.pygame_util.DrawOptions(screen)

#clock
clock = pygame.time.Clock()
FPS = 500

#game variables
lives = 1        # Number of times cue ball can be potted
dia = 36         # pixels
pocket_dia = 66  # pixels
active_force = 0
max_force = 20000
force_direction = 1
game_running = True
cue_ball_potted = False
taking_shot = True
powering_up = False
potted_balls = []

#colours
BG = (50, 50, 50)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

#fonts
font = pygame.font.SysFont("Lato", 30)
large_font = pygame.font.SysFont("Lato", 60)

#load images
cue_image = pygame.image.load("assets/images/cue.png").convert_alpha()
arrow_image_midpoint = pygame.image.load("assets/images/arrow_half.png").convert_alpha()
arrow_image_endpoint = pygame.image.load("assets/images/arrow.png").convert_alpha()
table_image = pygame.image.load("assets/images/table.png").convert_alpha()

# Load images of balls
ball_images = []
ball_image = pygame.image.load(f"assets/images/ball_1.png").convert_alpha()
cue_ball_image = pygame.image.load(f"assets/images/ball_16.png").convert_alpha()
ball_images.append(ball_image)
ball_images.append(cue_ball_image)

'''
for i in range(1, 17):
  ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()
  ball_images.append(ball_image)
'''

#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  screen.blit(img, (x, y))

def create_aim_boundary(radius, pos):
  body = pymunk.Body(body_type = pymunk.Body.STATIC)
  body.position = pos
  shape = pymunk.Circle(body, radius)
  shape.mass = 0
  shape.elasticity = 0.8
  #use pivot joint to add friction
  pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
  pivot.max_bias = 0.0 # disable joint correction
  pivot.max_force = 0 # emulate linear friction
  space.add(body, shape)
  return shape

#function for creating balls
def create_ball(radius, pos):
  body = pymunk.Body()
  body.position = pos
  shape = pymunk.Circle(body, radius)
  shape.mass = 5
  shape.elasticity = 0.8
  #use pivot joint to add friction
  pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
  pivot.max_bias = 0.0 # disable joint correction
  pivot.max_force = 1000 # emulate linear friction

  space.add(body, shape, pivot)
  return shape

#setup game balls
balls = []
columns = 5
rows = 5
#potting balls
col = 1
row = 1
pos = (250 + (col * (dia + 1)), 267 + (row * (dia + 1)) + (col * dia / 2))
new_ball = create_ball(dia / 2, pos)
balls.append(new_ball)

'''
for col in range(columns):
  for row in range(rows):
    pos = (250 + (col * (dia + 1)), 267 + (row * (dia + 1)) + (col * dia / 2))
    new_ball = create_ball(dia / 2, pos)
    balls.append(new_ball)
  rows -= 1
'''
#cue ball
pos = (888, SCREEN_HEIGHT / 2)
cue_ball = create_ball(dia / 2, pos)
balls.append(cue_ball)

#create six pockets on table
pockets = [
  (55, 63),
  (592, 48),
  (1134, 64),
  (55, 616),
  (592, 629),
  (1134, 616)
]

#create pool table cushions
cushions = [
  [(88, 56), (109, 77), (555, 77), (564, 56)],
  [(621, 56), (630, 77), (1081, 77), (1102, 56)],
  [(89, 621), (110, 600),(556, 600), (564, 621)],
  [(622, 621), (630, 600), (1081, 600), (1102, 621)],
  [(56, 96), (77, 117), (77, 560), (56, 581)],
  [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]
]

#function for creating cushions
def create_cushion(poly_dims):
  body = pymunk.Body(body_type = pymunk.Body.STATIC)
  body.position = ((0, 0))
  shape = pymunk.Poly(body, poly_dims)
  shape.elasticity = 0.8
  space.add(body, shape)

for c in cushions:
  create_cushion(c)

#create VectorArrow
class VectorArrow():
  def __init__(self, cue_ball_pos, vector_type = "midpoint"):
    if vector_type == "midpoint":
      self.original_image = arrow_image_midpoint
    elif type == "endpoint":
      self.original_image = arrow_image_endpoint
    self.angle = 0
    self.image = pygame.transform.rotate(self.original_image, self.angle)
    self.rect = self.image.get_rect()
    self.rect.center = cue_ball_pos

  def update(self, angle):
    self.angle = angle

  def draw(self, surface, new_arrow_width):
    # scale wrt original
    self.image = pygame.transform.scale(self.original_image, (new_arrow_width, self.original_image.get_height()))
    # rotate wrt self
    self.image = pygame.transform.rotate(self.image, self.angle)

    #self.rect = self.image.get_rect()
    #pygame.draw.rect(screen, BG, self.rect)
    for c in cushions:
      pygame.draw.polygon(screen, pygame.color.Color(0, 0, 255), c)

    for ball in balls:
      pygame.draw.circle(screen, pygame.color.Color(0, 0, 255), ball.body.position, dia / 2)

    for pocket in pockets:
      pygame.draw.circle(screen, pygame.color.Color(0, 0, 255), pocket, pocket_dia / 2)

    surface.blit(
      self.image,
      (self.rect.centerx - self.image.get_width() / 2,
      self.rect.centery - self.image.get_height() / 2)
      )

vector_arrow = VectorArrow(balls[-1].body.position)
loa_arrow = VectorArrow(balls[-1].body.position)
ray2_arrow = VectorArrow(balls[-2].body.position)
nearest_pocket_arrow = VectorArrow(balls[-2].body.position)

#create pool cue
class Cue():
  def __init__(self, pos):
    self.original_image = cue_image
    self.angle = 0
    self.image = pygame.transform.rotate(self.original_image, self.angle)
    self.rect = self.image.get_rect()
    self.rect.center = pos

  def update(self, angle):
    self.angle = angle

  def draw(self, surface):
    self.image = pygame.transform.rotate(self.original_image, self.angle)

    surface.blit(self.image,
      (self.rect.centerx - self.image.get_width() / 2, self.rect.centery - self.image.get_height() / 2)
     )

cue = Cue(balls[-1].body.position)

#create power bars to show how hard the cue ball will be hit
power_bar = pygame.Surface((10, 20))
power_bar.fill(RED)

justplayedimpactsound = True

winners = ["assets/audio/winner/winner1.mp3"]
losers = ["assets/audio/loser/loser1.mp3", "assets/audio/loser/loser2.mp3"]

#game loop
run = True
while run:
  clock.tick(FPS)
  space.step(1 / FPS)

  #fill background
  screen.fill(BG)

  #draw pool table
  screen.blit(table_image, (0, 0))

  #check if any balls have been potted
  for i, ball in enumerate(balls):
    for pocket in pockets:
      ball_x_dist = abs(ball.body.position[0] - pocket[0])
      ball_y_dist = abs(ball.body.position[1] - pocket[1])
      ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
      if ball_dist <= pocket_dia / 2:
        #check if the potted ball was the cue ball
        if i == len(balls) - 1:
          lives -= 1
          cue_ball_potted = True
          ball.body.position = (-100, -100)
          ball.body.velocity = (0.0, 0.0)
          pygame.mixer.music.load(f"assets/audio/loser/loser{random.randint(1, 6)}.mp3")
          pygame.mixer.music.play(0)
        else:
          space.remove(ball.body)
          balls.remove(ball)
          potted_balls.append(ball_images[i])
          ball_images.pop(i)
          pygame.mixer.music.load(f"assets/audio/winner/winner{random.randint(1, 2)}.mp3")
          pygame.mixer.music.play(0)

  #draw pool balls
  for i, ball in enumerate(balls):
    screen.blit(ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))

  #check if all the balls have stopped moving
  taking_shot = True
  for ball in balls:
    if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
      taking_shot = False

  #draw pool cue
  if taking_shot == True and game_running == True:
    space.remove(balls[0].body)
    aim_boundary = create_aim_boundary(dia, balls[0].body.position)

    if cue_ball_potted == True:
      #reposition cue ball
      balls[-1].body.position = (888, SCREEN_HEIGHT / 2)
      cue_ball_potted = False

    # Get positions of cursor, cue ball, and target ball
    mouse_pos = pygame.mouse.get_pos()
    #print(f"mouse_pos: {mouse_pos}")
    cue_ball_position = balls[-1].body.position
    target_ball_index = 0 # Will leave as the first ball in list for now
    target_ball_position = balls[target_ball_index].body.position

    #calculate pool cue angle
    cue.rect.center = cue_ball_position
    x_dist_cursor = mouse_pos[0] - cue_ball_position[0]
    y_dist_cursor = mouse_pos[1] - cue_ball_position[1] # -ve because pygame y coordinates increase down the screen
    cue_angle = math.degrees(math.atan2(y_dist_cursor, x_dist_cursor))
    cue.update(-cue_angle)
    cue.draw(screen)

    # Arrow to nearest pocket
    min_dist = 10000
    nearest_pocket = pockets[0]
    for pocket in pockets:
      x_dist_pocket1 = target_ball_position[0] - pocket[0]
      y_dist_pocket1 = -(target_ball_position[1] - pocket[1])
      euc_dist = math.sqrt(x_dist_pocket1*x_dist_pocket1 + y_dist_pocket1*y_dist_pocket1)
      if euc_dist < min_dist:
        min_dist = euc_dist
        nearest_pocket = pocket

    x_dist_pocket1 = target_ball_position[0] - nearest_pocket[0]
    y_dist_pocket1 = -(target_ball_position[1] - nearest_pocket[1])
    vector_angle = math.degrees(math.atan2(y_dist_pocket1, x_dist_pocket1))
    nearest_pocket_arrow.rect.center = target_ball_position
    new_arrow_width = 2 * math.sqrt(x_dist_pocket1*x_dist_pocket1 + y_dist_pocket1*y_dist_pocket1)
    nearest_pocket_arrow.update(vector_angle)
    nearest_pocket_arrow.draw(screen, new_arrow_width)

    # Check if LOA intersects target ball
    '''
    returns:
    - shape:  Shape that was hit, or None if no collision occured
    - point:  The point of impact.
    - normal: The normal of the surface hit.
    - alpha:  The normalized distance along the query segment in the range [0, 1]
    '''
    loa_angle = math.degrees(math.atan2(-y_dist_cursor, -x_dist_cursor)) # will be in the opp direction (mouse to cue ball)
    loa_angle_rad = np.deg2rad(loa_angle)
    loa_arrow.rect.center = [cue_ball_position[0] + dia*np.cos(loa_angle_rad), 
                             cue_ball_position[1] + dia*np.sin(loa_angle_rad)]
    
    loa_length = 2000
    start = loa_arrow.rect.center
    end = [start[0] + loa_length*np.cos(loa_angle_rad),
          start[1] + loa_length*np.sin(loa_angle_rad)]
    radius = 0.00001 # the radius of the object that is raycasted?
    filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS)
    raycast_info = space.segment_query_first(start=start, end=end, radius=radius, shape_filter=pymunk.ShapeFilter())
    if raycast_info != None:
      x_dist_loa = raycast_info.point[0] - start[0]
      y_dist_loa = raycast_info.point[1] - start[1]
    else:
      x_dist_loa = 2
      y_dist_loa = 2
    loa_arrow.update(-cue_angle)
    new_arrow_width = 2 * math.sqrt(x_dist_loa*x_dist_loa + y_dist_loa*y_dist_loa)
    loa_arrow.draw(screen, new_arrow_width)

    # Raycast 2
    ray2_arrow_width = 100
    if (raycast_info != None and type(raycast_info.shape) == pymunk.shapes.Circle):
      ray2_angle = math.degrees(np.arctan2(raycast_info.normal[1],raycast_info.normal[0]))
      ray2_angle_rad = np.deg2rad(ray2_angle)
      ray2_start = [raycast_info.point[0] - 2*dia*1.5*raycast_info.normal[0],
                    raycast_info.point[1] - 2*dia*1.5*raycast_info.normal[1]]
      ray2_arrow.rect.center = ray2_start
      ray2_end = [ray2_start[0] - loa_length*raycast_info.normal[0],
                  ray2_start[1] - loa_length*raycast_info.normal[1]]
      radius = 0.01 # the radius of the object that is raycasted
      filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS)
      raycast_info = space.segment_query_first(start=ray2_start, end=ray2_end, radius=radius, shape_filter=pymunk.ShapeFilter())
      if raycast_info != None:
        ray2_x_dist = raycast_info.point[0] - ray2_start[0]
        ray2_y_dist = raycast_info.point[1] - ray2_start[1]
      else:
        if abs(ray2_angle+vector_angle) < 0.01:
          powering_up == False and taking_shot == True
          active_force = 10000
        ray2_x_dist = 2000
        ray2_y_dist = 2000
      #print(f"ray2_angle: {ray2_angle}, vector_angle: {vector_angle}")
      ray2_arrow.update(-ray2_angle)
      ray2_arrow_width = 2 * math.sqrt(ray2_x_dist*ray2_x_dist + ray2_y_dist*ray2_y_dist)
      ray2_arrow.draw(screen, ray2_arrow_width)
    
    space.remove(aim_boundary)
    space.add(balls[0].body) 

  #power up pool cue
  if powering_up == True and game_running == True:
    active_force += 100 * force_direction
    if active_force >= max_force or active_force <= 0:
      force_direction *= -1
    #draw power bars
    for b in range(math.ceil(active_force / 2000)):
      screen.blit(power_bar,
       (balls[-1].body.position[0] - 30 + (b * 15),
        balls[-1].body.position[1] + 30))
  elif powering_up == False and taking_shot == True:
    x_impulse = math.cos(math.radians(-cue_angle))
    y_impulse = math.sin(math.radians(-cue_angle))
    balls[-1].body.apply_impulse_at_local_point((active_force * -x_impulse, active_force * y_impulse), (0, 0))
    active_force = 0
    force_direction = 1

  #draw bottom panel
  pygame.draw.rect(screen, BG, (0, SCREEN_HEIGHT, SCREEN_WIDTH, BOTTOM_PANEL))
  draw_text("LIVES: " + str(lives), font, WHITE, SCREEN_WIDTH - 200, SCREEN_HEIGHT + 10)
  
  #draw potted balls in bottom panel
  for i, ball in enumerate(potted_balls):
    screen.blit(ball, (10 + (i * 50), SCREEN_HEIGHT + 10))

  #check for game over
  if lives <= 0:
    draw_text("GAME OVER", large_font, WHITE, SCREEN_WIDTH / 2 - 160, SCREEN_HEIGHT / 2 - 100)
    game_running = False

  #check if all balls are potted
  if len(balls) == 1:
    draw_text("YOU WIN!", large_font, WHITE, SCREEN_WIDTH / 2 - 160, SCREEN_HEIGHT / 2 - 100)
    game_running = False

  #event handler
  for event in pygame.event.get():
    if event.type == pygame.MOUSEBUTTONDOWN and taking_shot == True:
      powering_up = True
    if event.type == pygame.MOUSEBUTTONUP and taking_shot == True:
      powering_up = False
    if event.type == pygame.QUIT:
      run = False

  #space.debug_draw(draw_options)
  pygame.display.update()

pygame.quit()

