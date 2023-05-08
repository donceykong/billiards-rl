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
  def __init__(self, pos):
    self.angle = 0
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

class BILLIARDS_GAME_HUMAN:
  def __init__(self):
    self.run = True
    self.reward = 0.0
    self.SCREEN_WIDTH = 1200
    self.SCREEN_HEIGHT = 678
    self.BOTTOM_PANEL = 50

    #game window
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
    self.max_force = 20000
    self.force_direction = 1
    self.game_running = True
    self.cue_ball_potted = False
    self.taking_shot = True
    self.powering_up = False
    self.potted_balls = []

    for i in range(1, 17):
      ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()
      self.ball_images.append(ball_image)
    
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
    
    #col = 0
    #row = 0
    #pos = (250 + (col * (self.dia + 1)), 267 + (row * (self.dia + 1)) + (col * self.dia / 2))
    #new_ball = self.create_ball(pos)
    #self.balls.append(new_ball)

    columns = 5
    rows = 5
    for col in range(columns):
      for row in range(rows):
        pos = (250 + (col * (self.dia + 1)), 267 + (row * (self.dia + 1)) + (col * self.dia / 2))
        new_ball = self.create_ball(pos)
        self.balls.append(new_ball)
      rows -= 1

    #cue ball
    pos = (888, self.SCREEN_HEIGHT / 2)
    cue_ball = self.create_ball(pos)
    self.balls.append(cue_ball)

    # Cushions
    for c in self.cushions:
      self.create_cushion(c)

    self.vector_arrow = VectorArrow(self.balls[-1].body.position)
    self.loa_arrow = VectorArrow(self.balls[-1].body.position)
    self.ray2_arrow = VectorArrow(self.balls[-2].body.position)
    self.nearest_pocket_arrow = VectorArrow(self.balls[-2].body.position)

    self.cue = Cue(self.balls[-1].body.position)
    #create power bars to show how hard the cue ball will be hit
    self.power_bar = self.pygame.Surface((10, 20))
    self.power_bar.fill(self.RED)
  
  def begin_human(self):
    #game loop
    while self.run:
      self.clock.tick(self.FPS)
      self.space.step(1 / self.FPS)
      self.screen.fill(self.BG)
      self.screen.blit(self.table_image, (0, 0))

      #check if any balls have been potted
      for i, ball in enumerate(self.balls):
        for pocket in self.pockets:
          ball_x_dist = abs(ball.body.position[0] - pocket[0])
          ball_y_dist = abs(ball.body.position[1] - pocket[1])
          ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
          if ball_dist <= self.pocket_dia / 2:
            #check if the potted ball was the cue ball
            if i == len(self.balls) - 1:
              self.lives -= 1
              cue_ball_potted = True
              ball.body.position = (-100, -100)
              ball.body.velocity = (0.0, 0.0)
              pygame.mixer.music.load(f"assets/audio/loser/loser{random.randint(1, 6)}.mp3")
              pygame.mixer.music.play(0)
            else:
              self.space.remove(ball.body)
              self.balls.remove(ball)
              self.potted_balls.append(self.ball_images[i])
              self.ball_images.pop(i)
              pygame.mixer.music.load(f"assets/audio/winner/winner{random.randint(1, 2)}.mp3")
              pygame.mixer.music.play(0)

      #draw pool balls
      for i, ball in enumerate(self.balls):
        self.screen.blit(self.ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))

      #check if all the balls have stopped moving
      self.taking_shot = True
      for ball in self.balls:
        if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
          self.taking_shot = False

      if self.taking_shot == True and self.game_running == True:
        target_ball = self.balls[-2].body
        target_ball_position = target_ball.position
        cue_ball = self.balls[-1].body
        cue_ball_position = cue_ball.position
        self.space.remove(target_ball)
        aim_boundary = self.create_aim_boundary(target_ball.position)

        if self.cue_ball_potted == True:
          #reposition cue ball
          target_ball.position = (888, self.SCREEN_HEIGHT / 2)
          self.cue_ball_potted = False

        # Get positions of cursor, cue ball, and target ball
        mouse_pos = pygame.mouse.get_pos()
        #print(f"mouse_pos: {mouse_pos}")

        #calculate pool cue angle
        x_dist_cursor = mouse_pos[0] - cue_ball_position[0]
        y_dist_cursor = mouse_pos[1] - cue_ball_position[1] # -ve because pygame y coordinates increase down the screen
        cue_angle = math.degrees(math.atan2(y_dist_cursor, x_dist_cursor))
        self.cue.update(-cue_angle)
        self.cue.rect.center = cue_ball_position
        self.cue.draw(self.screen)

        # Arrow to nearest pocket
        min_dist = 10000
        nearest_pocket = self.pockets[0]
        for pocket in self.pockets:
          x_dist_pocket1 = target_ball_position[0] - pocket[0]
          y_dist_pocket1 = -(target_ball_position[1] - pocket[1])
          euc_dist = math.sqrt(x_dist_pocket1*x_dist_pocket1 + y_dist_pocket1*y_dist_pocket1)
          if euc_dist < min_dist:
            min_dist = euc_dist
            nearest_pocket = pocket


        x_dist_pocket1 = target_ball_position[0] - nearest_pocket[0]
        y_dist_pocket1 = -(target_ball_position[1] - nearest_pocket[1])
        vector_angle = math.degrees(math.atan2(y_dist_pocket1, x_dist_pocket1))
        self.nearest_pocket_arrow.rect.center = target_ball_position
        new_arrow_width = 2 * math.sqrt(x_dist_pocket1*x_dist_pocket1 + y_dist_pocket1*y_dist_pocket1)
        self.nearest_pocket_arrow.update(vector_angle)
        self.nearest_pocket_arrow.draw(self.screen, new_arrow_width)

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
        loa_length = 2000
        start = [cue_ball_position[0] + self.dia*np.cos(loa_angle_rad), 
                cue_ball_position[1] + self.dia*np.sin(loa_angle_rad)]
        end = [start[0] + loa_length*np.cos(loa_angle_rad),
              start[1] + loa_length*np.sin(loa_angle_rad)]
        radius = 0.00001 # the radius of the object that is raycasted?
        raycast_info = self.space.segment_query_first(start=start, end=end, radius=radius, shape_filter=pymunk.ShapeFilter())
        if raycast_info != None:
          x_dist_loa = raycast_info.point[0] - start[0]
          y_dist_loa = raycast_info.point[1] - start[1]
        else:
          x_dist_loa = 2
          y_dist_loa = 2
        new_arrow_width = 2 * math.sqrt(x_dist_loa*x_dist_loa + y_dist_loa*y_dist_loa)
        

        self.loa_arrow.update(-cue_angle)
        self.loa_arrow.rect.center = start
        self.loa_arrow.draw(self.screen, new_arrow_width)

        # Raycast 2
        ray2_arrow_width = 100
        if (raycast_info != None and type(raycast_info.shape) == pymunk.shapes.Circle):
          ray2_angle = math.degrees(np.arctan2(raycast_info.normal[1],raycast_info.normal[0]))
          ray2_angle_rad = np.deg2rad(ray2_angle)
          ray2_start = [raycast_info.point[0] - self.dia*1.5*raycast_info.normal[0],
                        raycast_info.point[1] - self.dia*1.5*raycast_info.normal[1]]
          ray2_end = [ray2_start[0] - loa_length*raycast_info.normal[0],
                      ray2_start[1] - loa_length*raycast_info.normal[1]]
          radius = 0.01 # the radius of the object that is raycasted
          raycast_info = self.space.segment_query_first(start=ray2_start, end=ray2_end, radius=radius, shape_filter=pymunk.ShapeFilter())
          if raycast_info != None:
            ray2_x_dist = raycast_info.point[0] - ray2_start[0]
            ray2_y_dist = raycast_info.point[1] - ray2_start[1]
          else:
            if abs(ray2_angle+vector_angle) < 0.01:
              self.powering_up == False and self.taking_shot == True
              self.active_force = 10000
            ray2_x_dist = 2000
            ray2_y_dist = 2000
          #print(f"ray2_angle: {ray2_angle}, vector_angle: {vector_angle}")
          ray2_arrow_width = 2 * math.sqrt(ray2_x_dist*ray2_x_dist + ray2_y_dist*ray2_y_dist)
          self.ray2_arrow.update(-ray2_angle)
          self.ray2_arrow.rect.center = ray2_start
          self.ray2_arrow.draw(self.screen, ray2_arrow_width)
        self.space.remove(aim_boundary)
        self.space.add(target_ball) 

      #power up pool cue
      if self.powering_up == True and self.game_running == True:
        self.active_force += 100 * self.force_direction
        if self.active_force >= self.max_force or self.active_force <= 0:
          self.force_direction *= -1
        #draw power bars
        for b in range(math.ceil(self.active_force / 2000)):
          self.screen.blit(self.power_bar, (cue_ball_position[0] - 30 + (b * 15), cue_ball_position[1] + 30))
      elif self.powering_up == False and self.taking_shot == True:
        x_impulse = math.cos(math.radians(-cue_angle))
        y_impulse = math.sin(math.radians(-cue_angle))
        cue_ball.apply_impulse_at_local_point((self.active_force * -x_impulse, self.active_force * y_impulse), (0, 0))
        self.active_force = 0
        self.force_direction = 1

      #draw bottom panel
      pygame.draw.rect(self.screen, self.BG, (0, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.BOTTOM_PANEL))
      self.draw_text("LIVES: " + str(self.lives), self.font, self.WHITE, self.SCREEN_WIDTH - 200, self.SCREEN_HEIGHT + 10)
      #draw potted balls in bottom panel
      for i, ball in enumerate(self.potted_balls):
        self.screen.blit(ball, (10 + (i * 50), self.SCREEN_HEIGHT + 10))

      #check for game over
      if self.lives <= 0:
        self.draw_text("GAME OVER", self.large_font, self.WHITE, self.SCREEN_WIDTH / 2 - 160, self.SCREEN_HEIGHT / 2 - 100)
        self.game_running = False

      #check if all balls are potted
      if len(self.balls) == 1:
        self.draw_text("YOU WIN!", self.large_font, self.WHITE, self.SCREEN_WIDTH / 2 - 160, self.SCREEN_HEIGHT / 2 - 100)
        self.game_running = False

      #event handler
      for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and self.taking_shot == True:
          self.powering_up = True
        if event.type == pygame.MOUSEBUTTONUP and self.taking_shot == True:
          self.powering_up = False
        if event.type == pygame.QUIT:
          self.run = False

      self.space.debug_draw(self.draw_options)
      pygame.display.update()

    pygame.quit()