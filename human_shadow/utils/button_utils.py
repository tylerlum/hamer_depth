import pygame


class Button:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick found")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.reset()

    def reset(self):
        pygame.event.get()

    def is_pressed(self, up: bool=False) -> bool:
        for event in pygame.event.get():
            if up:
                if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    return True
            else:
                if event.type == pygame.JOYBUTTONDOWN:
                    return True
        return False

    def wait_for_press(self) -> bool:
        button_pressed = False
        while not button_pressed:
            button_pressed = self.is_pressed(up=True)
        pygame.event.get()
        return button_pressed

