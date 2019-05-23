import wiringpi
import pygame
import cv2 as cv

def gpioInit():
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(1, wiringpi.PWM_OUTPUT)
    wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
    wiringpi.pwmSetRange(12000)
    wiringpi.pwmSetClock(32)
    wiringpi.pinMode(22, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(23, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(24, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(25, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.softPwmCreate(22, 0, 100)
    wiringpi.softPwmCreate(23, 0, 100)
    wiringpi.softPwmCreate(24, 0, 100)
    wiringpi.softPwmCreate(25, 0, 100)


def setServoVal(val):
    if val < 380:
        val = 380
    if val > 620:
        val = 620
    wiringpi.pwmWrite(1, val)


def setMotorVal(val):
    if val >= 0:
        wiringpi.softPwmWrite(22, val)
        wiringpi.softPwmWrite(23, 0)
        wiringpi.softPwmWrite(24, 0)
        wiringpi.softPwmWrite(25, val)
    else:
        wiringpi.softPwmWrite(22, 0)
        wiringpi.softPwmWrite(23, val)
        wiringpi.softPwmWrite(24, val)
        wiringpi.softPwmWrite(25, 0)


def main():
    start_flag = 0
    speed = 0
    index = 0
    pygame.init()
    gpioInit()
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 240)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 60)
    screen = pygame.display.set_mode((0, 0), 0, 32)
    screen_rect = screen.get_rect()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("exit")
                    exit()
                elif event.key == pygame.K_w:
                    speed = speed + 1
                elif event.key == pygame.K_s:
                    speed = speed - 1
                else:
                    pass
        
                if start_flag:
                    setMotorVal(speed)
                else:
                    setMotorVal(0)
                print(speed)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                start_flag = ~start_flag
        x, y = pygame.mouse.get_pos()
        servo_val = 620 - int(x*240 / screen_rect.width)
        setServoVal(servo_val)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if start_flag:
            cv.imwrite('./image/' + str(index) + '_' + str(x), gray)
        index = index + 1
        wiringpi.delay(5)


if __name__ == "__main__":
    main()
