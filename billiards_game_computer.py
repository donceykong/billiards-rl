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
import time

#create VectorArrow
class VectorArrow():
  def __init__(self, cue_ball_pos, vector_type = "midpoint"):
    arrow_image_midpoint = pygame.image.load("assets/images/arrow_half.png").convert_alpha()
    arrow_image_endpoint = pygame.image.load("assets/images/arrow.png").convert_alpha()
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
    surface.blit(
      self.image,
      (self.rect.centerx - self.image.get_width() / 2,
      self.rect.centery - self.image.get_height() / 2)
    )

#create pool cue
class Cue():
  def __init__(self, pos, display=True):
    self.display = display
    self.angle = 0
    if self.display:
      cue_image = pygame.image.load("assets/images/cue.png").convert_alpha()
      self.original_image = cue_image
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

class BILLIARDS_GAME_COMPUTER:
  def __init__(self, cue_ball_states, target_ball_states, display = True, ):
    self.run = True
    self.reward = 0.0
    self.display = display
    self.SCREEN_WIDTH = 1200
    self.SCREEN_HEIGHT = 678
    self.BOTTOM_PANEL = 50
    self.cue_ball_states = cue_ball_states
    self.target_ball_states = target_ball_states
    
    #game window
    if self.display == True:
      self.pygame = pygame
      self.pygame.init()
      self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT + self.BOTTOM_PANEL))
      self.pygame.display.set_caption("Q_learning Billiards Display")
      self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
      self.BG = (50, 50, 50)
      self.RED = (255, 0, 0)
      self.WHITE = (255, 255, 255)
      self.font = pygame.font.SysFont("Lato", 30)
      self.large_font = pygame.font.SysFont("Lato", 60)
      #load images
      self.table_image = pygame.image.load("assets/images/table.png").convert_alpha()
      self.ball_images = []
      self.ball_image = pygame.image.load(f"assets/images/ball_1.png").convert_alpha()
      self.cue_ball_image = pygame.image.load(f"assets/images/ball_16.png").convert_alpha()
      self.ball_images.append(self.ball_image)
      self.ball_images.append(self.cue_ball_image)

    #pymunk space
    self.space = pymunk.Space()
    self.static_body = self.space.static_body

    #clock
    self.clock = pygame.time.Clock()
    self.FPS = 500

    #game variables
    self.lives = 1                # Number of times cue ball can be potted
    self.dia = 36                 # pixels
    self.pocket_dia = 66          # pixels
    self.active_force = 0
    self.max_force = 10000
    self.force_direction = 1
    self.game_running = True
    self.cue_ball_potted = False
    self.taking_shot = True
    self.powering_up = False
    self.potted_balls = []

    '''
    for i in range(1, 17):
      ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()
      ball_images.append(ball_image)
    '''
    
    #create pool table cushions
    self.cushions = [
      [(88, 56), (109, 77), (555, 77), (564, 56)],
      [(621, 56), (630, 77), (1081, 77), (1102, 56)],
      [(89, 621), (110, 600),(556, 600), (564, 621)],
      [(622, 621), (630, 600), (1081, 600), (1102, 621)],
      [(56, 96), (77, 117), (77, 560), (56, 581)],
      [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]
    ]

    #create six pockets on table
    self.pockets = [
      (55, 63),
      (592, 48),
      (1134, 64),
      (55, 616),
      (592, 629),
      (1134, 616)
    ]

  #function for outputting text onto center of the screen
  def draw_text(self, text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    self.screen.blit(img, (x, y))

  # Creates a fictitious object so that the 
  def create_aim_boundary(self, pos):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    body.position = pos
    shape = pymunk.Circle(body, self.dia)
    shape.mass = 0
    shape.elasticity = 0.8
    pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0)) #use pivot joint to add friction
    pivot.max_bias = 0.0 # disable joint correction
    pivot.max_force = 0 # emulate linear friction
    self.space.add(body, shape)
    return shape

  #function for creating balls
  def create_ball(self, pos):
    body = pymunk.Body()
    body.position = pos
    radius = self.dia / 2
    shape = pymunk.Circle(body, radius)
    shape.mass = 5
    shape.elasticity = 0.8
    pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0)) #use pivot joint to add friction
    pivot.max_bias = 0.0 # disable joint correction
    pivot.max_force = 1000 # emulate linear friction
    self.space.add(body, shape, pivot)
    return shape
  
  #function for creating cushions
  def create_cushion(self, poly_dims):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    body.position = ((0, 0))
    shape = pymunk.Poly(body, poly_dims)
    shape.elasticity = 0.8
    self.space.add(body, shape)
      
  def setup_game(self):
    #setup game balls
    self.balls = []
    #potting balls
    col = 0
    row = 0
    pos = (250 + (col * (self.dia + 1)), 267 + (row * (self.dia + 1)) + (col * self.dia / 2))
    new_ball = self.create_ball(pos)
    self.balls.append(new_ball)

    '''
    columns = 5
    rows = 5
    for col in range(columns):
      for row in range(rows):
        pos = (250 + (col * (dia + 1)), 267 + (row * (dia + 1)) + (col * dia / 2))
        new_ball = create_ball(dia / 2, pos)
        balls.append(new_ball)
      rows -= 1
    '''

    #cue ball
    pos = (888, self.SCREEN_HEIGHT / 2)
    cue_ball = self.create_ball(pos)
    self.balls.append(cue_ball)

    # Cushions
    for c in self.cushions:
      self.create_cushion(c)

    if self.display == True:
      self.vector_arrow = VectorArrow(self.balls[-1].body.position)
      self.loa_arrow = VectorArrow(self.balls[-1].body.position)
      self.ray2_arrow = VectorArrow(self.balls[-2].body.position)
      self.nearest_pocket_arrow = VectorArrow(self.balls[-2].body.position)

    self.cue = Cue(self.balls[-1].body.position, self.display)

    #create power bars to show how hard the cue ball will be hit
    if self.display == True:
      self.power_bar = pygame.Surface((10, 20))
      self.power_bar.fill(self.RED)

  def shoot(self, input_angle, input_power):
    self.taking_shot = False
    self.reward = 0
    has_velocity = True
    cue_angle = input_angle
    self.cue.update(-cue_angle)
    #check if the potted ball was the cue ball
    #self.balls[-1].body.position = [int(self.balls[-1].body.position[0]), int(self.balls[-1].body.position[1])]
    #self.balls[-2].body.position = [int(self.balls[-2].body.position[0]), int(self.balls[-2].body.position[1])]
    self.balls[-1].body.position = self.cue_ball_states[np.random.randint(len(self.cue_ball_states))]
    self.balls[-2].body.position = self.target_ball_states[np.random.randint(len(self.target_ball_states))]
    target_ball_position_og = self.balls[-2].body.position
    cue_ball_position_og = self.balls[-1].body.position

    if self.display == True:
      self.cue.rect.center = self.balls[-1].body.position
      self.cue.draw(self.screen)
    if self.game_running == True:
      x_impulse = math.cos(math.radians(-cue_angle))
      y_impulse = math.sin(math.radians(-cue_angle))
      self.balls[-1].body.apply_impulse_at_local_point((input_power * -x_impulse, input_power * y_impulse), (0, 0)) 
    
    while(has_velocity):
      self.clock.tick(self.FPS)
      self.space.step(1 / self.FPS)
      if self.display == True:
        self.screen.fill(self.BG)
        self.screen.blit(self.table_image, (0, 0))
      #check if any balls have been potted
      for i, ball in enumerate(self.balls):
        for pocket in self.pockets:
          ball_x_dist = abs(ball.body.position[0] - pocket[0])
          ball_y_dist = abs(ball.body.position[1] - pocket[1])
          ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
          if ball_dist <= self.pocket_dia / 2:
            if i == len(self.balls) - 1:
              self.lives -= 1
              self.cue_ball_potted = True
              ball.body.position = (-100, -100)
              ball.body.velocity = (0.0, 0.0)
              self.run = False
              self.reward = -100
            else:
              ball.body.position = (-100, -100)
              ball.body.velocity = (0.0, 0.0)
              #self.space.remove(ball.body)
              #self.balls.remove(ball)
              self.run = False
              self.reward = 100
              #if self.display == True:
                #self.potted_balls.append(self.ball_images[i])
                #self.ball_images.pop(i)
      #draw pool balls
      if self.display == True:
        for i, ball in enumerate(self.balls):
          self.screen.blit(self.ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))
      #draw bottom panel
      if self.display == True:
        pygame.draw.rect(self.screen, self.BG, (0, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.BOTTOM_PANEL))
        self.draw_text("LIVES: " + str(self.lives), self.font, self.WHITE, self.SCREEN_WIDTH - 200, self.SCREEN_HEIGHT + 10)
        #draw potted balls in bottom panel
        for i, ball in enumerate(self.potted_balls):
          self.screen.blit(ball, (10 + (i * 50), self.SCREEN_HEIGHT + 10))
      #check for game over
      if self.lives <= 0:
        if self.display == True:
          self.draw_text("GAME OVER", self.large_font, self.WHITE, self.SCREEN_WIDTH / 2 - 160, self.SCREEN_HEIGHT / 2 - 100)
        self.game_running = False
        self.run = False
        self.reward = -100
      #check if all balls are potted
      if len(self.balls) == 1 and self.lives != 0:
        if self.display == True:
          self.draw_text("YOU WIN!", self.large_font, self.WHITE, self.SCREEN_WIDTH / 2 - 160, self.SCREEN_HEIGHT / 2 - 100)
        self.game_running = False
        self.run = False
        self.reward = 100
      #check if all the balls have stopped moving
      cue_vel = self.balls[-1].body.velocity
      target_vel = self.balls[-2].body.velocity
      cue_speed = np.sqrt(np.power(cue_vel[0],2) + np.power(cue_vel[1],2))
      target_speed = np.sqrt(np.power(target_vel[0],2) + np.power(target_vel[1],2))
      if int(cue_speed) != 0 or int(target_speed) != 0:
        has_velocity = True
      if int(cue_speed) == 0 and int(target_speed) == 0:
        has_velocity = False
        self.reward += -10
      if self.display == True:
        #self.space.debug_draw(self.draw_options)
        pygame.display.update()
      
    return cue_ball_position_og, target_ball_position_og, self.reward